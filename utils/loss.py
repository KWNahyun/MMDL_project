import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistillationLosses(nn.Module):
    """
    1단계 지식 증류를 위한 복합 손실 함수.
    InfoNCE (CLIP), Similarity-KD, Post-Cosine Loss를 포함합니다.
    """
    def __init__(self, temperature=0.07, kd_temp=4.0):
        super().__init__()
        # InfoNCE Loss (CLIP)의 Temperature 학습 파라미터
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        
        # Similarity-KD의 Temperature (Hyperparameter)
        self.kd_temp = kd_temp

    def get_logits(self, img_emb, txt_emb):
        """
        L2 정규화된 임베딩과 Logit Scale을 사용하여 유사도 행렬(logits)을 계산합니다.
        """
        # L2 정규화
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(txt_emb, dim=-1)

        # Logit Scale 적용
        scale = self.logit_scale.exp()
        logits_i = scale * img @ txt.t()
        logits_t = logits_i.t()
        
        return logits_i, logits_t

    def clip_loss(self, logits_i, logits_t):
        """
        InfoNCE Loss (L_clip)를 계산합니다.
        """
        N = logits_i.size(0)
        labels = torch.arange(N, device=logits_i.device)
        
        loss_i = F.cross_entropy(logits_i, labels)
        loss_t = F.cross_entropy(logits_t, labels)
        
        return (loss_i + loss_t) / 2
        
    def similarity_kd_loss(self, logits_teacher, logits_student):
        """
        Similarity-KD Loss (L_sim)를 계산합니다 (KL Divergence 기반).
        교사 모델과 학생 모델의 로짓 분포를 정렬합니다.
        """
        # 교사의 Softmax 분포 (높은 온도 적용)
        # logits_teacher는 이미 정규화된 embedding의 내적이므로 scale이 작을 수 있음
        # 보통 KD에서는 logit 자체를 사용하지만, 여기서는 cosine similarity map을 사용하므로
        # Teacher의 분포를 Softmax로 만들어 Target으로 사용
        
        # KLDivLoss: Input은 LogSoftmax, Target은 Softmax (또는 확률분포)
        soft_target = F.softmax(logits_teacher / self.kd_temp, dim=-1)
        soft_student = F.log_softmax(logits_student / self.kd_temp, dim=-1)
        
        # KL Divergence 계산 (Batchmean reduction 권장)
        kd_loss = F.kl_div(soft_student, soft_target, reduction='batchmean')
        
        # 온도의 제곱을 곱하여 그라디언트 스케일 보정
        return kd_loss * (self.kd_temp ** 2)

    def post_cosine_loss(self, img_emb_teacher, img_emb_student):
        """
        Post-Feature Cosine Loss (L_cos)를 계산합니다.
        학생 이미지 임베딩을 교사 이미지 임베딩 방향에 정렬합니다.
        """
        # 코사인 유사도 (Dim=-1은 특징 차원)
        # F.cosine_similarity는 입력이 이미 정규화되어 있지 않아도 내부적으로 처리하지만,
        # 명시적으로 정규화된 입력을 주어도 무방함.
        cos_sim = F.cosine_similarity(img_emb_teacher, img_emb_student, dim=-1)
        
        # 코사인 유사도를 최대화 = (1 - Cosine Similarity)를 최소화
        return 1.0 - cos_sim.mean()

    def forward(self, img_emb_student, txt_emb_teacher, img_emb_teacher, logits_teacher, weights=None):
        """
        전체 증류 손실 L_total을 계산합니다.
        
        Args:
            img_emb_student: 학생 모델 이미지 임베딩 (Gradient O)
            txt_emb_teacher: 교사 모델 텍스트 임베딩 (Gradient X)
            img_emb_teacher: 교사 모델 이미지 임베딩 (Gradient X, L_cos용)
            logits_teacher:  교사 모델의 Logit 행렬 (Gradient X, L_sim용)
            weights: Loss별 가중치 딕셔너리
        """
        if weights is None:
            weights = {'w_clip': 1.0, 'w_sim': 0.0, 'w_cos': 0.0}

        # 1. InfoNCE Loss (L_clip)
        # 학생 이미지 <-> 교사 텍스트 간의 Contrastive Learning
        logits_s_i, logits_s_t = self.get_logits(img_emb_student, txt_emb_teacher)
        L_clip = self.clip_loss(logits_s_i, logits_s_t)

        # 2. Post-Feature Cosine Loss (L_cos)
        # 학생 이미지 임베딩이 교사 이미지 임베딩을 모사하도록 함
        if weights.get('w_cos', 0) > 0:
            L_cos = self.post_cosine_loss(img_emb_teacher, img_emb_student)
        else:
            L_cos = torch.tensor(0.0, device=img_emb_student.device)
        
        # 3. Similarity-KD Loss (L_sim)
        # 학생의 (Image x Text) 관계 맵이 교사의 관계 맵을 닮도록 함
        if weights.get('w_sim', 0) > 0:
            # 학생의 Logit (Scale 적용 전 순수 내적값 사용 권장 or Scale 포함 등 구현에 따라 다름)
            # 여기서는 get_logits 내부에서 Scale이 적용되므로, SimKD를 위해
            # Scale 적용 전의 순수 Cosine Map을 다시 계산하거나, get_logits 반환값을 활용할 수 있음.
            # 논문 구현체에 따라 다르나, 보통 Temperature가 적용된 Logits을 비교함.
            
            # 학생의 Logits (Text Encoder는 Teacher 것이므로 고정)
            # z_s @ z_t.T
            img_s_norm = F.normalize(img_emb_student, dim=-1)
            txt_t_norm = F.normalize(txt_emb_teacher, dim=-1)
            logits_student = img_s_norm @ txt_t_norm.t()
            
            L_sim = self.similarity_kd_loss(logits_teacher, logits_student)
        else:
            L_sim = torch.tensor(0.0, device=img_emb_student.device)

        # 4. 최종 Loss 합산
        L_total = (weights.get('w_clip', 1.0) * L_clip) + \
                  (weights.get('w_cos', 0.0) * L_cos) + \
                  (weights.get('w_sim', 0.0) * L_sim)
                  
        return L_total, L_clip, L_cos, L_sim