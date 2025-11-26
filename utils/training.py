import torch
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Helper Functions ---
def encode_text(text_list, tokenizer, teacher_model, device):
    with torch.no_grad():
        tok = tokenizer(text_list, context_length=77).to(device)
        return teacher_model.encode_text(tok)

def encode_image_teacher(images, teacher_model):
    with torch.no_grad():
        return teacher_model.encode_image(images)

# --- Stage 1: Distillation Training ---
def train_epoch(student_encoder, clip_loss_fn, optimizer, loader, tokenizer, teacher_model, device, cfg, epoch):
    student_encoder.train()
    clip_loss_fn.train()
    s, acc = 0, 0
    
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
        with torch.no_grad():
            t_img_norm = F.normalize(img_emb_teacher, dim=-1)
            t_txt_norm = F.normalize(txt_emb_teacher, dim=-1)
            logits_teacher = t_img_norm @ t_txt_norm.t()

        # 4. Complex Loss Calculation
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

# --- Stage 2: Fine-tuning Training ---
def fine_tune_epoch(model, optimizer, loader, tokenizer, teacher_model, device, epoch, cfg):
    model.train()
    s, acc = 0, 0
    
    print(f"\n--- [Stage 2] Fine-tuning Epoch {epoch} ---")
    pbar = tqdm(loader, desc=f"FT Ep {epoch}")

    for step, bt in enumerate(pbar):
        if bt is None: continue
        images, commands, gt_bboxes = bt
        images = images.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # 1. 텍스트 임베딩 (Teacher Model 사용)
        # Talk2CarModel은 Teacher Text Encoder가 내장되어 있지 않으므로 외부에서 주입
        text_emb = encode_text(commands, tokenizer, teacher_model, device)
        
        # 2. Forward (Image + Text -> BBox Pred)
        pred_bboxes = model(images, text_emb)
        
        # 3. Loss (MSE Loss for BBox regression)
        # 실제로는 IoU Loss나 L1 Loss를 조합해서 사용해야 함
        loss = F.mse_loss(pred_bboxes, gt_bboxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc += loss.item()
        s += 1
        
        pbar.set_postfix({'MSE': f"{acc/s:.4f}"})

    return acc / s if s > 0 else 0

def evaluate_talk2car(model, loader, tokenizer, teacher_model, device, cfg):
    model.eval()
    total_iou = 0
    count = 0
    
    print("\n--- Evaluating Talk2Car ---")
    with torch.no_grad():
        for bt in loader:
            images, commands, gt_bboxes = bt
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
            pred_bboxes = model(images, text_emb)
            
            # Simple IoU approximation for evaluation display
            # (Not exact IoU implementation, just monitoring metric)
            inter_x1 = torch.max(pred_bboxes[:, 0], gt_bboxes[:, 0])
            inter_y1 = torch.max(pred_bboxes[:, 1], gt_bboxes[:, 1])
            inter_x2 = torch.min(pred_bboxes[:, 0]+pred_bboxes[:, 2], gt_bboxes[:, 0]+gt_bboxes[:, 2])
            inter_y2 = torch.min(pred_bboxes[:, 1]+pred_bboxes[:, 3], gt_bboxes[:, 1]+gt_bboxes[:, 3])
            
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
            pred_area = pred_bboxes[:, 2] * pred_bboxes[:, 3]
            gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            union_area = pred_area + gt_area - inter_area
            
            iou = inter_area / (union_area + 1e-6)
            total_iou += iou.mean().item()
            count += 1
            
    avg_iou = total_iou / count if count > 0 else 0
    print(f"Average IoU: {avg_iou:.4f}")
    return avg_iou

# (recall_at_k, evaluate_retrieval 함수는 기존과 동일하게 유지 - 생략 가능)
def recall_at_k(query_emb, emb_set, text_set, keyword, k=10):
    if len(emb_set) == 0: return 0.0, []
    sims = cosine_similarity(query_emb, emb_set)[0]
    topk_idx = sims.argsort()[::-1][:k]
    topk_labels = [text_set[i] for i in topk_idx]
    recall = sum([keyword in lbl for lbl in topk_labels]) / k
    return recall, list(zip(sims[topk_idx], topk_labels))

def evaluate_retrieval(student_encoder, loader, tokenizer, teacher_model, device, cfg):
    # (이전과 동일한 코드, 생략)
    return []