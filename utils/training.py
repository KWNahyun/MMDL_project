import torch
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================================================================
# Helper Functions
# ==============================================================================

def encode_text(text_list, tokenizer, teacher_model, device):
    """Teacher Text Encoder를 사용하여 텍스트 임베딩을 계산합니다 (Frozen)."""
    with torch.no_grad():
        tok = tokenizer(text_list, context_length=77).to(device)
        return teacher_model.encode_text(tok)

def encode_image_teacher(images, teacher_model):
    """Teacher Image Encoder를 사용하여 이미지 임베딩을 계산합니다 (Frozen)."""
    with torch.no_grad():
        return teacher_model.encode_image(images)

# ==============================================================================
# Stage 1: Knowledge Distillation (Alignment)
# ==============================================================================

def train_epoch(student_encoder, clip_loss_fn, optimizer, loader, tokenizer, teacher_model, device, cfg, epoch):
    """
    1단계 지식 증류를 위한 단일 훈련 에포크를 실행합니다.
    InfoNCE, Similarity-KD, Post-Cosine Loss를 복합적으로 사용합니다.
    """
    student_encoder.train()
    clip_loss_fn.train()
    s, acc = 0, 0
    
    # Config에서 가중치 가져오기
    weights = cfg['TRAIN']['LOSS_WEIGHTS']
    
    print(f"\n--- [Stage 1] Epoch {epoch} (Weights: {weights}) ---")
    pbar = tqdm(loader, desc=f"Ep {epoch}")

    for step, bt in enumerate(pbar):
        if bt is None: continue
        images, texts = bt
        images = images.to(device)

        # 1. Teacher Outputs (Frozen)
        txt_emb_teacher = encode_text(texts, tokenizer, teacher_model, device)
        img_emb_teacher = encode_image_teacher(images, teacher_model)
        
        # 2. Student Output (Trainable)
        img_emb_student = student_encoder(images)

        # 3. Teacher Logits (for SimKD)
        # Teacher의 Image-Text 관계(Logit)를 계산하여 Student가 이를 모사하도록 함
        with torch.no_grad():
            t_img_norm = F.normalize(img_emb_teacher, dim=-1)
            t_txt_norm = F.normalize(txt_emb_teacher, dim=-1)
            logits_teacher = t_img_norm @ t_txt_norm.t()

        # 4. Complex Loss Calculation
        # (InfoNCE + SimKD + PostCosine)
        L_total, L_clip, L_cos, L_sim = clip_loss_fn(
            img_emb_student=img_emb_student,
            txt_emb_teacher=txt_emb_teacher,
            img_emb_teacher=img_emb_teacher,
            logits_teacher=logits_teacher,
            weights=weights
        )

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        acc += L_total.item()
        s += 1
        
        pbar.set_postfix({'Loss': f"{acc/s:.4f}", 'CLIP': f"{L_clip.item():.2f}"})

    return acc / s if s > 0 else 0

# ==============================================================================
# Stage 2: Talk2Car Fine-tuning (Grounding)
# ==============================================================================

def fine_tune_epoch(model, loss_fn, optimizer, loader, tokenizer, teacher_model, device, epoch, cfg):
    """
    2단계 Talk2Car 파인튜닝을 위한 단일 훈련 에포크를 실행합니다.
    L1 Loss (좌표 거리) + GIoU Loss (박스 겹침 최적화)를 사용합니다.
    """
    model.train()
    loss_fn.train() 
    s, acc = 0, 0
    
    # Config에서 가중치 가져오기
    weights = cfg['TALK2CAR']['FINE_TUNE']['LOSS_WEIGHTS']
    
    print(f"\n--- [Stage 2] Fine-tuning Epoch {epoch} (L1:{weights['w_l1']}, GIoU:{weights['w_giou']}) ---")
    pbar = tqdm(loader, desc=f"FT Ep {epoch}")

    for step, bt in enumerate(pbar):
        if bt is None: continue
        images, commands, gt_bboxes = bt
        images = images.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # 1. Text Embedding (Teacher Text Encoder 사용)
        with torch.no_grad():
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
        
        # 2. Forward Pass (Image + Text -> BBox Prediction)
        pred_bboxes = model(images, text_emb)
        
        # 3. Loss Calculation (L1 + GIoU)
        L_total, L_l1, L_giou = loss_fn(pred_bboxes, gt_bboxes, weights)

        # 4. Backward
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        acc += L_total.item()
        s += 1
        
        # 로그에 세부 Loss 표시
        pbar.set_postfix({'Total': f"{acc/s:.3f}", 'L1': f"{L_l1.item():.3f}", 'GIoU': f"{L_giou.item():.3f}"})

    return acc / s if s > 0 else 0

def evaluate_talk2car(model, loader, tokenizer, teacher_model, device, cfg):
    """
    Talk2Car 데이터셋에 대해 모델을 평가하고 평균 IoU 및 AP50을 계산합니다.
    """
    model.eval()
    total_iou = 0
    total_correct_05 = 0 # AP50 (IoU >= 0.5) 측정을 위한 카운터
    total_samples = 0
    
    print("\n--- Evaluating Talk2Car ---")
    with torch.no_grad():
        for bt in tqdm(loader, desc="Evaluating"):
            images, commands, gt_bboxes = bt
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
            pred_bboxes = model(images, text_emb)
            
            # IoU 계산 (Batch 단위)
            # Box format: [x, y, w, h] (Normalized 0~1)
            
            # [x1, y1, x2, y2]로 변환
            pred_x1 = pred_bboxes[:, 0]
            pred_y1 = pred_bboxes[:, 1]
            pred_x2 = pred_bboxes[:, 0] + pred_bboxes[:, 2]
            pred_y2 = pred_bboxes[:, 1] + pred_bboxes[:, 3]
            
            gt_x1 = gt_bboxes[:, 0]
            gt_y1 = gt_bboxes[:, 1]
            gt_x2 = gt_bboxes[:, 0] + gt_bboxes[:, 2]
            gt_y2 = gt_bboxes[:, 1] + gt_bboxes[:, 3]

            # Intersection 영역 계산
            inter_x1 = torch.max(pred_x1, gt_x1)
            inter_y1 = torch.max(pred_y1, gt_y1)
            inter_x2 = torch.min(pred_x2, gt_x2)
            inter_y2 = torch.min(pred_y2, gt_y2)
            
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            
            # Union 영역 계산
            pred_area = pred_bboxes[:, 2] * pred_bboxes[:, 3]
            gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            union_area = pred_area + gt_area - inter_area
            
            # IoU 벡터 계산
            iou = inter_area / (union_area + 1e-6)
            
            # 통계 누적
            total_iou += iou.sum().item()
            total_correct_05 += (iou >= 0.5).sum().item()
            total_samples += images.size(0)
            
    avg_iou = total_iou / total_samples if total_samples > 0 else 0
    ap50 = (total_correct_05 / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"AP50 (IoU >= 0.5): {ap50:.2f}%") # 리더보드 비교용 지표
    
    return avg_iou, ap50

# ==============================================================================
# Stage 1 Evaluation Utilities
# ==============================================================================

def recall_at_k(query_emb, emb_set, text_set, keyword, k=10):
    """
    주어진 쿼리 임베딩에 대해 임베딩 세트에서 Recall@K를 계산합니다.
    """
    if len(emb_set) == 0: return 0.0, []
    
    # 코사인 유사도 계산
    sims = cosine_similarity(query_emb, emb_set)[0]
    
    # 상위 K개 인덱스 추출
    topk_idx = sims.argsort()[::-1][:k]
    topk_labels = [text_set[i] for i in topk_idx]
    
    # Recall 계산 (키워드가 포함된 항목 수 / K)
    recall = sum([keyword in lbl for lbl in topk_labels]) / k
    return recall, list(zip(sims[topk_idx], topk_labels))

def evaluate_retrieval(student_encoder, loader, tokenizer, teacher_model, device, cfg):
    """
    영역 은행(Region Bank)을 구축하고 크기별(Small/Large) Recall@K를 분석합니다.
    """
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
    
    # 텍스트 기반으로 Small/Large 영역 분리 (COCO Region 텍스트 특성 활용)
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
        
        # 쿼리 임베딩
        q_emb = encode_text([q_small_text], tokenizer, teacher_model, device)
        q_emb = F.normalize(q_emb, dim=-1).cpu().numpy()

        r_small, _ = recall_at_k(q_emb, emb_small, text_small, keyword=cls, k=k)
        r_large, _ = recall_at_k(q_emb, emb_large, text_large, keyword=cls, k=k)

        results.append((cls, len([t for t in text_small if cls in t]), len([t for t in text_large if cls in t]), r_small, r_large))

    # 결과 요약 출력
    print("\n===== SUMMARY (class-wise small vs large) =====")
    print(f"{'class':13s} | {'#small':>7s} | {'#large':>7s} | {'SmallR@10':>9s} | {'LargeR@10':>9s}")
    for cls, n_s, n_l, rs, rl in results:
        print(f"{cls:13s} | {n_s:7d} | {n_l:7d} | {rs:9.3f} | {rl:9.3f}")

    return results