# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DistillationLosses(nn.Module):
    """
    Stage 1 지식 증류 복합 손실
    InfoNCE (CLIP) + Similarity-KD + Post-Cosine
    """
    def __init__(self, temperature=0.07, kd_temp=4.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        self.kd_temp = kd_temp

    def get_logits(self, img_emb, txt_emb):
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(txt_emb, dim=-1)
        scale = self.logit_scale.exp()
        logits_i = scale * img @ txt.t()
        logits_t = logits_i.t()
        return logits_i, logits_t

    def clip_loss(self, logits_i, logits_t):
        N = logits_i.size(0)
        labels = torch.arange(N, device=logits_i.device)
        loss_i = F.cross_entropy(logits_i, labels)
        loss_t = F.cross_entropy(logits_t, labels)
        return (loss_i + loss_t) / 2
        
    def similarity_kd_loss(self, logits_teacher, logits_student):
        soft_target = F.softmax(logits_teacher / self.kd_temp, dim=-1)
        soft_student = F.log_softmax(logits_student / self.kd_temp, dim=-1)
        kd_loss = F.kl_div(soft_student, soft_target, reduction='batchmean')
        return kd_loss * (self.kd_temp ** 2)

    def post_cosine_loss(self, img_emb_teacher, img_emb_student):
        cos_sim = F.cosine_similarity(img_emb_teacher, img_emb_student, dim=-1)
        return 1.0 - cos_sim.mean()

    def forward(self, img_emb_student, txt_emb_teacher, img_emb_teacher, logits_teacher, weights=None):
        if weights is None:
            weights = {'w_clip': 1.0, 'w_sim': 0.0, 'w_cos': 0.0}

        # InfoNCE
        logits_s_i, logits_s_t = self.get_logits(img_emb_student, txt_emb_teacher)
        L_clip = self.clip_loss(logits_s_i, logits_s_t)

        # Post-Cosine
        if weights.get('w_cos', 0) > 0:
            L_cos = self.post_cosine_loss(img_emb_teacher, img_emb_student)
        else:
            L_cos = torch.tensor(0.0, device=img_emb_student.device)
        
        # Similarity-KD
        if weights.get('w_sim', 0) > 0:
            img_s_norm = F.normalize(img_emb_student, dim=-1)
            txt_t_norm = F.normalize(txt_emb_teacher, dim=-1)
            logits_student = img_s_norm @ txt_t_norm.t()
            L_sim = self.similarity_kd_loss(logits_teacher, logits_student)
        else:
            L_sim = torch.tensor(0.0, device=img_emb_student.device)

        L_total = (weights.get('w_clip', 1.0) * L_clip) + \
                  (weights.get('w_cos', 0.0) * L_cos) + \
                  (weights.get('w_sim', 0.0) * L_sim)
                  
        return L_total, L_clip, L_cos, L_sim


class CIoULoss(nn.Module):
    """
    Complete IoU Loss
    기존 GIoU보다 수렴이 빠르고 정확하며, Aspect Ratio(종횡비)를 고려하여
    작은 객체의 형태를 더 잘 잡아냅니다.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [B, 4] (x_top_left, y_top_left, w, h)
            target_boxes: [B, 4] (x_top_left, y_top_left, w, h)
        """
        # 1. 좌표 변환: (x, y, w, h) -> (x1, y1, x2, y2)
        b1_x1, b1_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
        b1_w, b1_h = pred_boxes[:, 2], pred_boxes[:, 3]
        b1_x2, b1_y2 = b1_x1 + b1_w, b1_y1 + b1_h

        b2_x1, b2_y1 = target_boxes[:, 0], target_boxes[:, 1]
        b2_w, b2_h = target_boxes[:, 2], target_boxes[:, 3]
        b2_x2, b2_y2 = b2_x1 + b2_w, b2_y1 + b2_h

        # 2. Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # 3. Union
        w1, h1 = b1_w, b1_h
        w2, h2 = b2_w, b2_h
        union_area = w1 * h1 + w2 * h2 - inter_area + self.eps

        # 4. IoU
        iou = inter_area / union_area

        # 5. Enclosing Box (최소 외접 사각형)의 대각선 길이 (c^2)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + self.eps

        # 6. 중심점 거리 (rho^2)
        b1_cx, b1_cy = b1_x1 + w1 / 2, b1_y1 + h1 / 2
        b2_cx, b2_cy = b2_x1 + w2 / 2, b2_y1 + h2 / 2
        rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

        # 7. Aspect Ratio 보정 (v)
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + self.eps)) - torch.atan(w1 / (h1 + self.eps)), 2)

        # 8. Alpha
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # 9. CIoU Loss
        ciou = iou - (rho2 / c2) - alpha * v
        loss = 1.0 - ciou

        return loss.mean()


class Talk2CarLoss(nn.Module):
    """
    Stage 2 Fine-tuning Loss
    L1 + CIoU (Updated from GIoU)
    """
    def __init__(self, lambda_l1=5.0, lambda_ciou=2.0):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ciou_loss = CIoULoss()
        
        # 기본 가중치 (외부에서 weights 딕셔너리가 오지 않을 경우 사용)
        self.default_l1_w = lambda_l1
        self.default_ciou_w = lambda_ciou

    def forward(self, pred_bbox, gt_bbox, weights=None):
        """
        Args:
            pred_bbox: [B, 4] normalized (x, y, w, h)
            gt_bbox: [B, 4] normalized (x, y, w, h)
            weights: dict, optional (e.g. {'w_l1': 5.0, 'w_giou': 2.0})
                     * 호환성을 위해 키값이 'w_giou'여도 내부적으로 CIoU 가중치로 씁니다.
        """
        # 1. L1 Loss
        loss_l1 = self.l1_loss(pred_bbox, gt_bbox)

        # 2. CIoU Loss
        loss_ciou = self.ciou_loss(pred_bbox, gt_bbox)

        # 3. 가중치 적용
        w_l1 = self.default_l1_w
        w_ciou = self.default_ciou_w
        
        if weights is not None:
            w_l1 = weights.get('w_l1', w_l1)
            # 기존 코드와의 호환성을 위해 'w_giou' 키가 있으면 그걸 CIoU 가중치로 사용
            if 'w_giou' in weights:
                w_ciou = weights['w_giou']
            elif 'w_ciou' in weights:
                w_ciou = weights['w_ciou']

        loss_total = (w_l1 * loss_l1) + (w_ciou * loss_ciou)
        
        return loss_total, loss_l1, loss_ciou


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced data
    Talk2Car에서 특정 명령어가 과도하게 많을 때 사용
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()