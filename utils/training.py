# utils/training.py
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import open_clip
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.visualization import save_visualization
from pathlib import Path
import json

# ==============================================================================
# Helper Functions
# ==============================================================================

def encode_text(text_list, tokenizer, teacher_model, device):
    """Teacher Text Encoder (Frozen)"""
    with torch.no_grad():
        tok = tokenizer(text_list, context_length=77).to(device)
        return teacher_model.encode_text(tok)

def encode_image_teacher(images, teacher_model):
    """Teacher Image Encoder (Frozen)"""
    with torch.no_grad():
        return teacher_model.encode_image(images)

# ==============================================================================
# Stage 0: Teacher Domain Adaptation (NEW)
# ==============================================================================

def adapt_teacher_to_talk2car(teacher_model, tokenizer, t2c_dir, device, cfg):
    """
    Stage 0: Teacher를 Talk2Car 도메인에 적응
    - Talk2Car 이미지-명령어 쌍으로 Contrastive Learning
    - Visual Encoder만 학습
    """
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torch.optim as optim
    
    print("\n=== Stage 0: Teacher Domain Adaptation ===")
    
    # Visual Encoder만 학습
    for name, param in teacher_model.named_parameters():
        if "visual" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Talk2Car Caption Dataset
    class Talk2CarCaptionDataset(Dataset):
        def __init__(self, data_dir, split, transform):
            self.data_dir = Path(data_dir)
            self.transform = transform
            
            json_file = self.data_dir / "commands" / f"{split}_commands.json"
            with open(json_file, 'r') as f:
                content = json.load(f)
            
            self.samples = []
            img_dir = self.data_dir / "images"
            
            for item in content["commands"]:
                img_path = img_dir / item['t2c_img']
                if img_path.exists():
                    self.samples.append((img_path, item['command']))
            
            print(f"   Loaded {len(self.samples)} samples for adaptation")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, command = self.samples[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                return self.transform(img), command
            except:
                # Fallback
                return self.transform(Image.new('RGB', (224, 224))), command
    
    # Transform (OpenCLIP 표준)
    _, _, preprocess = open_clip.create_model_and_transforms(
        cfg['TEACHER_MODEL'], pretrained=None
    )
    
    # Dataset & Loader
    train_ds = Talk2CarCaptionDataset(t2c_dir, 'train', preprocess)
    
    # KITTI 스타일 오버샘플링은 Talk2Car에선 불필요 (충분히 큼)
    loader = DataLoader(train_ds, 
                       batch_size=cfg['TEACHER_ADAPTATION']['BATCH_SIZE'], 
                       shuffle=True, num_workers=4)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, teacher_model.parameters()),
        lr=float(cfg['TEACHER_ADAPTATION']['LEARNING_RATE'])
    )
    
    # Training Loop
    teacher_model.train()
    num_epochs = cfg['TEACHER_ADAPTATION']['NUM_EPOCHS']
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Teacher Adaptation Ep{epoch+1}/{num_epochs}")
        
        for imgs, texts in pbar:
            imgs = imgs.to(device)
            text_tokens = tokenizer(texts).to(device)
            
            # CLIP Contrastive Loss
            img_features = teacher_model.encode_image(imgs)
            text_features = teacher_model.encode_text(text_tokens)
            
            img_features = F.normalize(img_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            logit_scale = teacher_model.logit_scale.exp()
            logits_per_image = logit_scale * img_features @ text_features.T
            logits_per_text = logits_per_image.T
            
            labels = torch.arange(len(imgs), device=device)
            loss = (F.cross_entropy(logits_per_image, labels) +
                   F.cross_entropy(logits_per_text, labels)) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"   Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
    
    # Freeze again
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_model.eval()
    print("✅ Teacher adaptation completed")
    
    return teacher_model


def train_epoch(student_encoder, clip_loss_fn, optimizer, loader, tokenizer, teacher_model, device, cfg, epoch):
    student_encoder.train()
    clip_loss_fn.train()
    
    # Scaler 초기화
    scaler = GradScaler()
    
    s, acc = 0, 0
    weights = cfg['TRAIN']['LOSS_WEIGHTS']
    
    # [NEW] 그래디언트 누적 스텝 설정 (Config에 없으면 기본값 4 사용)
    # 배치 사이즈를 1/4로 줄였으면 accumulation_steps=4로 설정하여 원래 효과 유지
    accumulation_steps = cfg['TRAIN'].get('ACCUMULATION_STEPS', 4)
    
    pbar = tqdm(loader, desc=f"Ep {epoch}")

    # [수정 1] 루프 시작 전에 그래디언트 초기화
    optimizer.zero_grad()

    for step, bt in enumerate(pbar):
        if bt is None: continue
        images, texts = bt
        images = images.to(device)

        # 1. Teacher (FP16 Context)
        with torch.no_grad():
            with autocast():
                txt_emb_teacher = encode_text(texts, tokenizer, teacher_model, device)
                img_emb_teacher = encode_image_teacher(images, teacher_model)
                
                t_img_norm = F.normalize(img_emb_teacher, dim=-1)
                t_txt_norm = F.normalize(txt_emb_teacher, dim=-1)
                logits_teacher = t_img_norm @ t_txt_norm.t()

        # 2. Student 학습
        # [삭제] optimizer.zero_grad() -> 여기서 매번 초기화하면 안 됨 (누적해야 하므로)
        
        with autocast():
            img_emb_student = student_encoder(images)
            
            L_total, L_clip, L_cos, L_sim = clip_loss_fn(
                img_emb_student=img_emb_student,
                txt_emb_teacher=txt_emb_teacher,
                img_emb_teacher=img_emb_teacher,
                logits_teacher=logits_teacher,
                weights=weights
            )
            
            # [수정 2] Loss를 누적 스텝 수로 나누기 (평균을 맞추기 위함)
            L_total = L_total / accumulation_steps

        # 3. Backward (Scale)
        scaler.scale(L_total).backward()

        # [수정 3] 정해진 스텝마다 파라미터 업데이트 (Accumulation)
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # 업데이트 후에 초기화

        # 로깅 (나눴던 값을 다시 곱해서 원래 스케일로 표시)
        current_loss = L_total.item() * accumulation_steps
        acc += current_loss
        s += 1
        
        pbar.set_postfix({'Loss': f"{acc/s:.4f}"})
        
    # 남은 그래디언트 처리 (선택 사항: 루프가 끝났는데 업데이트 안 된 게 있으면 처리)
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return acc / s if s > 0 else 0
# ==============================================================================
# Stage 2: Talk2Car Fine-tuning
# ==============================================================================

def fine_tune_epoch(model, loss_fn, optimizer, loader, tokenizer, teacher_model, device, epoch, cfg):
    """Stage 2 훈련 에포크"""
    model.train()
    loss_fn.train() 
    s, acc = 0, 0
    
    weights = cfg['TALK2CAR']['FINE_TUNE']['LOSS_WEIGHTS']
    
    print(f"\n--- [Stage 2] Fine-tuning Epoch {epoch} (L1:{weights['w_l1']}, GIoU:{weights['w_giou']}) ---")
    pbar = tqdm(loader, desc=f"FT Ep {epoch}")

    for step, bt in enumerate(pbar):
        images, commands, gt_bboxes, _ = bt  # command_token 무시
        images = images.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # Text Embedding
        with torch.no_grad():
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
        
        # Forward
        pred_bboxes = model(images, text_emb)
        
        # Loss
        L_total, L_l1, L_giou = loss_fn(pred_bboxes, gt_bboxes, weights)

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        acc += L_total.item()
        s += 1
        
        pbar.set_postfix({'Total': f"{acc/s:.3f}", 'L1': f"{L_l1.item():.3f}", 'GIoU': f"{L_giou.item():.3f}"})

    return acc / s if s > 0 else 0

def evaluate_talk2car(model, loader, tokenizer, teacher_model, device, cfg):
    """Talk2Car Validation 평가"""
    model.eval()
    total_iou = 0
    total_correct_05 = 0
    total_samples = 0
    
    print("\n--- Evaluating Talk2Car ---")
    with torch.no_grad():
        for bt in tqdm(loader, desc="Evaluating"):
            images, commands, gt_bboxes, _ = bt
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
            pred_bboxes = model(images, text_emb)
            
            # IoU 계산
            pred_x1 = pred_bboxes[:, 0]
            pred_y1 = pred_bboxes[:, 1]
            pred_x2 = pred_bboxes[:, 0] + pred_bboxes[:, 2]
            pred_y2 = pred_bboxes[:, 1] + pred_bboxes[:, 3]
            
            gt_x1 = gt_bboxes[:, 0]
            gt_y1 = gt_bboxes[:, 1]
            gt_x2 = gt_bboxes[:, 0] + gt_bboxes[:, 2]
            gt_y2 = gt_bboxes[:, 1] + gt_bboxes[:, 3]

            inter_x1 = torch.max(pred_x1, gt_x1)
            inter_y1 = torch.max(pred_y1, gt_y1)
            inter_x2 = torch.min(pred_x2, gt_x2)
            inter_y2 = torch.min(pred_y2, gt_y2)
            
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            
            pred_area = pred_bboxes[:, 2] * pred_bboxes[:, 3]
            gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            union_area = pred_area + gt_area - inter_area
            
            iou = inter_area / (union_area + 1e-6)
            
            total_iou += iou.sum().item()
            total_correct_05 += (iou >= 0.5).sum().item()
            total_samples += images.size(0)
            
    avg_iou = total_iou / total_samples if total_samples > 0 else 0
    ap50 = (total_correct_05 / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"AP50 (IoU >= 0.5): {ap50:.2f}%")
    
    return avg_iou, ap50

# ==============================================================================
# TEST INFERENCE: predictions.json 생성 (NEW)
# ==============================================================================

def generate_predictions_json(model, test_loader, tokenizer, teacher_model, device, save_path, cfg):
    """
    Talk2Car Test set에 대한 predictions.json 생성
    
    Format: {command_token: [x0, y0, w, h]}
    - Absolute coordinates (pixels)
    - Talk2Car 이미지 크기: 1600 x 900
    """
    model.eval()
    predictions = {}
    
    print(f"\n🚀 Generating predictions.json for Talk2Car test set...")
    
    # Talk2Car 표준 이미지 크기
    IMG_WIDTH = 1600
    IMG_HEIGHT = 900
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            images, commands, _, command_tokens = batch
            images = images.to(device)
            
            # Text Embedding
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
            
            # Prediction (Normalized [0~1])
            pred_bboxes = model(images, text_emb)  # [B, 4]
            
            # Denormalize to absolute coordinates
            for i, token in enumerate(command_tokens):
                x_norm, y_norm, w_norm, h_norm = pred_bboxes[i].cpu().numpy()
                
                # Normalized -> Absolute
                x0 = int(x_norm * IMG_WIDTH)
                y0 = int(y_norm * IMG_HEIGHT)
                w = int(w_norm * IMG_WIDTH)
                h = int(h_norm * IMG_HEIGHT)
                
                # Clamp to image bounds
                x0 = max(0, min(x0, IMG_WIDTH - 1))
                y0 = max(0, min(y0, IMG_HEIGHT - 1))
                w = max(1, min(w, IMG_WIDTH - x0))
                h = max(1, min(h, IMG_HEIGHT - y0))
                
                predictions[token] = [x0, y0, w, h]
    
    # Save JSON
    with open(save_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Predictions saved to {save_path}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Format: {{command_token: [x0, y0, w, h]}}")
    print(f"\n📤 Ready for submission to Talk2Car leaderboard!")
    print(f"   URL: https://eval.ai/web/challenges/challenge-page/835/overview")
    
    return predictions

def inference_and_visualize(model, loader, tokenizer, teacher_model, device, save_dir, max_vis=50):
    """Test 셋 시각화"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Running Inference on Test Set (Saving {max_vis} visualizations) ---")
    
    total_iou = 0
    total_correct_05 = 0
    count = 0
    vis_count = 0
    
    with torch.no_grad():
        for i, bt in enumerate(tqdm(loader, desc="Testing")):
            images, commands, gt_bboxes, _ = bt
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
            pred_bboxes = model(images, text_emb)
            
            for j in range(images.size(0)):
                p_box = pred_bboxes[j]
                g_box = gt_bboxes[j]
                
                # IoU 계산
                p_x1, p_y1 = p_box[0], p_box[1]
                p_x2, p_y2 = p_box[0] + p_box[2], p_box[1] + p_box[3]
                
                g_x1, g_y1 = g_box[0], g_box[1]
                g_x2, g_y2 = g_box[0] + g_box[2], g_box[1] + g_box[3]
                
                inter_x1 = torch.max(p_x1, g_x1)
                inter_y1 = torch.max(p_y1, g_y1)
                inter_x2 = torch.min(p_x2, g_x2)
                inter_y2 = torch.min(p_y2, g_y2)
                
                inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
                pred_area = p_box[2] * p_box[3]
                gt_area = g_box[2] * g_box[3]
                union_area = pred_area + gt_area - inter_area
                
                iou = (inter_area / (union_area + 1e-6)).item()
                
                total_iou += iou
                if iou >= 0.5:
                    total_correct_05 += 1
                count += 1
                
                # 시각화
                if vis_count < max_vis:
                    clean_cmd = "".join(c for c in commands[j] if c.isalnum())[:20]
                    fname = f"vis_{vis_count:03d}_iou_{iou:.2f}_{clean_cmd}.jpg"
                    
                    save_visualization(
                        images[j], 
                        commands[j], 
                        p_box.cpu(), 
                        g_box.cpu(), 
                        save_dir / fname,
                        iou
                    )
                    vis_count += 1

    avg_iou = total_iou / count if count > 0 else 0
    ap50 = (total_correct_05 / count) * 100 if count > 0 else 0
    
    print(f"\n[Test Result] Average IoU: {avg_iou:.4f}")
    print(f"[Test Result] AP50 (IoU >= 0.5): {ap50:.2f}%")
    print(f"[Visualization] Saved {vis_count} images to {save_dir}")
    
    return avg_iou, ap50

# ==============================================================================
# Stage 1 Evaluation
# ==============================================================================

def recall_at_k(query_emb, emb_set, text_set, keyword, k=10):
    """Recall@K 계산"""
    if len(emb_set) == 0: return 0.0, []
    
    sims = cosine_similarity(query_emb, emb_set)[0]
    topk_idx = sims.argsort()[::-1][:k]
    topk_labels = [text_set[i] for i in topk_idx]
    
    recall = sum([keyword in lbl for lbl in topk_labels]) / k
    return recall, list(zip(sims[topk_idx], topk_labels))

def evaluate_retrieval(student_encoder, loader, tokenizer, teacher_model, device, cfg):
    """Region Bank 기반 Retrieval 평가"""
    student_encoder.eval()
    teacher_model.eval()

    region_embs = []
    region_texts = []
    MAX_BATCHES = cfg['EVAL']['MAX_BANK_BATCHES']

    print("\n--- Building Region Bank for Retrieval Evaluation ---")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if batch is None: continue
            images, texts = batch

            images = images.to(device)
            img_emb = student_encoder(images)
            img_emb = F.normalize(img_emb, dim=-1)

            region_embs.append(img_emb.cpu())
            region_texts.extend(texts)

            if i >= MAX_BATCHES: break

    if not region_embs:
        print("Region bank is empty. Skipping evaluation.")
        return []

    region_embs = torch.cat(region_embs, dim=0)
    print(f"Region bank size: {region_embs.shape[0]} regions")

    embs_np = region_embs.numpy()
    
    def is_small(t: str):
        ts = t.lower()
        return ("very small" in ts) or ("a small" in ts)

    def is_large(t: str):
        ts = t.lower()
        return ("a large" in ts)

    idx_small = [i for i, txt in enumerate(region_texts) if is_small(txt)]
    idx_large = [i for i, txt in enumerate(region_texts) if is_large(txt)]

    emb_small = embs_np[idx_small]
    emb_large = embs_np[idx_large]
    
    text_small = [region_texts[i] for i in idx_small]
    text_large = [region_texts[i] for i in idx_large]
    
    print(f"Small regions: {len(idx_small)}, Large regions: {len(idx_large)}")

    results = []
    k = cfg['EVAL']['TOP_K']
    target_classes = cfg['TARGET_CLASS_NAMES']

    print(f"\nEvaluating Recall@{k} on target classes...")
    for cls in target_classes:
        q_small_text = f"a small {cls}"
        
        q_emb = encode_text([q_small_text], tokenizer, teacher_model, device)
        q_emb = F.normalize(q_emb, dim=-1).cpu().numpy()

        r_small, _ = recall_at_k(q_emb, emb_small, text_small, keyword=cls, k=k)
        r_large, _ = recall_at_k(q_emb, emb_large, text_large, keyword=cls, k=k)

        results.append((cls, len([t for t in text_small if cls in t]), len([t for t in text_large if cls in t]), r_small, r_large))

    print("\n===== SUMMARY (class-wise small vs large) =====")
    print(f"{'class':13s} | {'#small':>7s} | {'#large':>7s} | {'SmallR@10':>9s} | {'LargeR@10':>9s}")
    for cls, n_s, n_l, rs, rl in results:
        print(f"{cls:13s} | {n_s:7d} | {n_l:7d} | {rs:9.3f} | {rl:9.3f}")

    return results
