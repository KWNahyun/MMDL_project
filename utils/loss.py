# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class Talk2CarLoss(nn.Module):
    """
    Stage 2 Fine-tuning Loss
    L1 + GIoU
    """
    def __init__(self):
        super().__init__()

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c), (y_c), (x_c + w), (y_c + h)]
        return torch.stack(b, dim=-1)

    def generalized_box_iou(self, boxes1, boxes2):
        # Intersection
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter

        # IoU
        iou = inter / (union + 1e-6)

        # Enclosing Box
        lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        wh_c = (rb_c - lt_c).clamp(min=0)
        area_c = wh_c[:, 0] * wh_c[:, 1]

        # GIoU
        giou = iou - ((area_c - union) / (area_c + 1e-6))
        return giou

    def forward(self, pred_bbox, gt_bbox, weights):
        # L1 Loss
        loss_l1 = F.l1_loss(pred_bbox, gt_bbox, reduction='mean')

        # GIoU Loss
        pred_xyxy = self.box_cxcywh_to_xyxy(pred_bbox)
        gt_xyxy = self.box_cxcywh_to_xyxy(gt_bbox)
        
        giou = self.generalized_box_iou(pred_xyxy, gt_xyxy)
        loss_giou = 1.0 - giou.mean()

        loss_total = (weights['w_l1'] * loss_l1) + (weights['w_giou'] * loss_giou)
        
        return loss_total, loss_l1, loss_giou


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced data (팀원 코드 반영)
    Talk2Car에서 특정 명령어가 과도하게 많을 때 사용
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 클래스별 가중치 [num_classes]
        self.gamma = gamma  # Hard example focusing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
