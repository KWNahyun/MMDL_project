# models/student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from models.grounding_head import ProjectionGroundingHead, TransformerGroundingHead

class DistilledConvNeXtTiny(nn.Module):
    """
    기본 Student 모델 (Single-Scale)
    Stage 1: Global Feature 출력
    Stage 2: Spatial Feature 제공
    """
    def __init__(self, text_dim, backbone_name="convnext_tiny"):
        super().__init__()
        # Spatial Feature 유지
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='')
        
        # Stage 1 Distillation용 Global Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.num_features, text_dim)

    def forward(self, x):
        """Stage 1: Distillation용 (Global Feature)"""
        feat_map = self.backbone(x)  # [B, C, H, W]
        pooled = self.pool(feat_map).flatten(1)  # [B, C]
        emb = self.head(pooled)
        return emb


class DistilledConvNeXtTinyMultiScale(nn.Module):
    """
    Multi-Scale Student 모델 (팀원 코드 반영)
    - Stage 2, 3, 4를 모두 추출하여 Concat
    - 다양한 크기의 객체 특징 포착
    """
    def __init__(self, text_dim, backbone_name="convnext_tiny"):
        super().__init__()
        # Stage 2, 3, 4 추출 (features_only=True)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=True, 
            features_only=True, 
            out_indices=(1, 2, 3)  # Stage 2, 3, 4
        )
        
        # Feature 차원: Stage2=192, Stage3=384, Stage4=768 → Total=1344
        self.fusion = nn.Sequential(
            nn.Conv2d(1344, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1 Distillation용 Global Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, text_dim)
        
        # Feature 차원 저장
        self.num_features = 256

    def forward(self, x):
        """Stage 1: Distillation용 (Global Feature)"""
        c2, c3, c4 = self.backbone(x)
        h, w = c2.shape[-2:]
        
        # 모든 스케일을 Stage2 크기로 맞춤
        c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate & Fuse
        fused = self.fusion(torch.cat([c2, c3, c4], dim=1))  # [B, 256, H, W]
        
        # Global Pooling
        pooled = self.pool(fused).flatten(1)  # [B, 256]
        return self.head(pooled)
    
    def get_spatial_features(self, x):
        """
        Stage 2: Grounding용 (Spatial Feature 유지)
        Talk2CarModel에서 호출
        """
        c2, c3, c4 = self.backbone(x)
        h, w = c2.shape[-2:]
        
        c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        
        return self.fusion(torch.cat([c2, c3, c4], dim=1))  # [B, 256, H, W]


class Talk2CarModel(nn.Module):
    """
    Stage 2 파인튜닝 통합 모델
    - Distilled Encoder + Grounding Head
    - Config에 따라 Head Type 선택 (Projection / Transformer)
    """
    def __init__(self, distilled_encoder, text_dim, head_type):
        super().__init__()
        
        self.image_encoder = distilled_encoder
        
        # Encoder의 Feature 차원 확인
        if hasattr(distilled_encoder, 'num_features'):
            encoder_feat_dim = distilled_encoder.num_features
        else:
            encoder_feat_dim = distilled_encoder.backbone.num_features
        
        self.head_type = head_type
        print(f"[Model] Initializing Talk2CarModel with Head Type: {head_type}")

        # Grounding Head 선택
        if head_type == "Transformer":
            self.grounding_head = TransformerGroundingHead(
                image_feat_dim=encoder_feat_dim,
                text_dim=text_dim,
                hidden_dim=256,
                nhead=4,
                num_layers=3
            )
        elif head_type == "Projection":
            self.grounding_head = ProjectionGroundingHead(
                image_feat_dim=encoder_feat_dim,
                text_dim=text_dim
            )
        else:
            raise ValueError(f"Unknown HEAD_TYPE: {head_type}")
            
    def forward(self, images, text_emb):
        """
        Forward Pass
        Args:
            images: [B, 3, H, W]
            text_emb: [B, text_dim]
        Returns:
            pred_bbox: [B, 4] (normalized x, y, w, h)
        """
        # 1. Image Feature 추출
        # Multi-Scale 모델이면 get_spatial_features 사용
        if hasattr(self.image_encoder, 'get_spatial_features'):
            image_feat = self.image_encoder.get_spatial_features(images)
        else:
            # 기본 모델은 backbone 직접 호출
            image_feat = self.image_encoder.backbone(images)
        
        # 2. Grounding Head 통과
        pred_bbox = self.grounding_head(image_feat, text_emb)
        
        return pred_bbox


def load_student_encoder(path, text_dim, backbone_name, device):
    """
    Student Encoder 로드
    - backbone_name에 따라 Single-Scale / Multi-Scale 선택
    """
    # Multi-Scale 옵션 확인
    if "multiscale" in backbone_name.lower():
        print("[Model] Using Multi-Scale ConvNeXt-Tiny")
        base_backbone = backbone_name.replace("_multiscale", "")
        model = DistilledConvNeXtTinyMultiScale(text_dim=text_dim, backbone_name=base_backbone)
    else:
        print("[Model] Using Single-Scale ConvNeXt-Tiny")
        model = DistilledConvNeXtTiny(text_dim=text_dim, backbone_name=backbone_name)
    
    # 가중치 로드
    if path and os.path.exists(path):
        print(f"[Model] Loading student weights from: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        
        # Strict=False로 로드 (Head 크기 불일치 허용)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[Warning] Missing keys: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")
    else:
        print(f"[Model] No checkpoint found at '{path}'. Using ImageNet pretrained backbone.")
        
    return model.to(device)
