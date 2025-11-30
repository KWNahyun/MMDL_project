import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from models.grounding_head import ProjectionGroundingHead, TransformerGroundingHead

class DistilledConvNeXtTiny(nn.Module):
    """
    기본 Student 모델 (Single-Scale)
    - Stage 1: Global Feature 출력 (Contrastive Learning용)
    - Stage 2: Spatial Feature 출력 (Grounding용, No Pooling)
    """
    def __init__(self, text_dim, backbone_name="convnext_tiny"):
        super().__init__()
        # [핵심 1] global_pool='' 설정으로 Pooling을 비활성화하여 (B, C, H, W) 맵 유지
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='')
        
        # Stage 1 Distillation을 위한 Global Pooling 및 Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.num_features, text_dim)

    def forward(self, x):
        """Stage 1: Distillation용 (Global Feature)"""
        feat_map = self.backbone(x)  # [B, C, H, W]
        
        # [핵심 2] Stage 1에서는 Crop된 Region 이미지가 들어오므로, 
        # 이를 Global Pooling 해도 결과적으로는 "Local Region Feature"가 됨.
        pooled = self.pool(feat_map).flatten(1)  # [B, C]
        emb = self.head(pooled)
        return emb
    
    def get_spatial_features(self, x):
        """Stage 2: Grounding용 (Spatial Feature)"""
        # Pooling 없이 (B, C, H, W) 맵을 그대로 반환 -> Transformer Head가 좌표를 찾음
        return self.backbone(x)


# class DistilledConvNeXtTinyMultiScale(nn.Module):
#     """
#     Multi-Scale Student 모델
#     - 작은 객체와 큰 객체를 모두 잘 잡기 위해 여러 계층의 Feature를 융합
#     - ConvNeXt의 Stage 2, 3, 4 출력을 결합
#     """
#     def __init__(self, text_dim, backbone_name="convnext_tiny"):
#         super().__init__()
#         # features_only=True: 중간 레이어의 Feature Map들을 리스트로 반환
#         self.backbone = timm.create_model(
#             backbone_name, 
#             pretrained=True, 
#             features_only=True, 
#             out_indices=(1, 2, 3)  # Stage 2(1/8), 3(1/16), 4(1/32) 추출
#         )
        
#         # Feature 차원 융합 (1x1 Conv)
#         # Stage 2(192) + Stage 3(384) + Stage 4(768) = 1344 채널
#         # 이를 256 채널로 압축하여 Transformer 입력으로 사용
#         self.fusion = nn.Sequential(
#             nn.Conv2d(1344, 256, kernel_size=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
        
#         # Stage 1용 Head
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.head = nn.Linear(256, text_dim)
        
#         self.num_features = 256

#     def _get_fused_features(self, x):
#         """Feature 추출 및 Multi-scale Fusion 공통 로직"""
#         # c2: 1/8 scale, c3: 1/16 scale, c4: 1/32 scale
#         c2, c3, c4 = self.backbone(x)
#         h, w = c2.shape[-2:] # 가장 큰 해상도(c2) 기준
        
#         # Upsampling하여 크기 맞춤 (Interpolation)
#         c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
#         c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        
#         # 채널 방향으로 결합 (Concat)
#         concat_feat = torch.cat([c2, c3, c4], dim=1) # [B, 1344, H, W]
        
#         # 채널 압축 (Fusion)
#         fused_feat = self.fusion(concat_feat) # [B, 256, H, W]
#         return fused_feat

#     def forward(self, x):
#         """Stage 1: Distillation용 (Global Feature)"""
#         fused = self._get_fused_features(x) # [B, 256, H, W]
#         pooled = self.pool(fused).flatten(1) # [B, 256]
#         return self.head(pooled)
    
#     def get_spatial_features(self, x):
#         """Stage 2: Grounding용 (Spatial Feature) - No Pooling"""
#         return self._get_fused_features(x) # [B, 256, H, W] 그대로 반환


class DistilledConvNeXtTinyMultiScale(nn.Module):
    def __init__(self, text_dim, backbone_name="convnext_tiny"):
        super().__init__()
        # [수정 1] out_indices에 0번(Stride 4) 추가 -> 작은 객체 정보 확보
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=True, 
            features_only=True, 
            out_indices=(0, 1, 2, 3)
        )
        
        # 채널 정의 (ConvNeXt-Tiny 기준: 96, 192, 384, 768)
        dims = self.backbone.feature_info.channels() # [96, 192, 384, 768]
        
        # 각 스케일별 투영 레이어 (모두 256 채널로 통일)
        self.proj_c0 = nn.Conv2d(dims[0], 256, kernel_size=3, padding=1, stride=2) # Stride 4 -> 8 (Downsample)
        self.proj_c1 = nn.Conv2d(dims[1], 256, kernel_size=1)                      # Stride 8 -> 8 (Keep)
        self.proj_c2 = nn.Conv2d(dims[2], 256, kernel_size=1)                      # Stride 16 -> 8 (Upsample)
        self.proj_c3 = nn.Conv2d(dims[3], 256, kernel_size=1)                      # Stride 32 -> 8 (Upsample)
        
        # [수정 2] CoordConv용 추가 채널 (256 + 2 = 258)
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4 + 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Refinement
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, text_dim)
        self.num_features = 256

    def _get_fused_features(self, x):
        # c0: 1/4, c1: 1/8, c2: 1/16, c3: 1/32
        c0, c1, c2, c3 = self.backbone(x)
        
        # 1. 모든 피처를 Stride 8 (c1 크기)로 맞춤
        # c0 (Stride 4) -> Convolution으로 줄임 (정보 압축)
        f0 = self.proj_c0(c0) 
        
        # c1 (Stride 8) -> 그대로
        f1 = self.proj_c1(c1)
        
        # c2, c3 -> Interpolation으로 늘림
        h, w = c1.shape[-2:]
        f2 = F.interpolate(self.proj_c2(c2), size=(h, w), mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.proj_c3(c3), size=(h, w), mode='bilinear', align_corners=False)
        
        # 2. Concat
        x_feat = torch.cat([f0, f1, f2, f3], dim=1) # [B, 1024, H, W]

        # [수정 3] Coordinate Channel 추가 (CoordConv)
        # 모델이 "여기가 왼쪽 위다, 오른쪽 아래다"를 명시적으로 알게 해줌
        batch_size, _, height, width = x_feat.shape
        
        xx_channel = torch.arange(width, dtype=x_feat.dtype, device=x_feat.device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        yy_channel = torch.arange(height, dtype=x_feat.dtype, device=x_feat.device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        # Normalize to [-1, 1]
        xx_channel = (xx_channel / (width - 1)) * 2 - 1
        yy_channel = (yy_channel / (height - 1)) * 2 - 1
        
        x_feat = torch.cat([x_feat, xx_channel, yy_channel], dim=1) # [B, 1026, H, W]
        
        # 3. Final Fusion
        return self.fusion(x_feat)

    def forward(self, x):
        fused = self._get_fused_features(x)
        pooled = self.pool(fused).flatten(1)
        return self.head(pooled)
    
    def get_spatial_features(self, x):
        return self._get_fused_features(x)


class Talk2CarModel(nn.Module):
    """
    Stage 2 파인튜닝 통합 모델
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

        # [핵심 3] Cross-Attention 기반 Transformer Head 사용 권장
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
        Args:
            images: [B, 3, H, W]
            text_emb: [B, text_dim]
        """
        # 1. Image Feature 추출 (Spatial Feature: [B, C, H, W])
        # get_spatial_features 메서드가 있으면 사용 (Multi-Scale 등)
        if hasattr(self.image_encoder, 'get_spatial_features'):
            image_feat = self.image_encoder.get_spatial_features(images)
        else:
            # 기본 모델 (get_spatial_features를 위에서 구현했으므로 실제론 여기 안 탐)
            image_feat = self.image_encoder.backbone(images)
        
        # 2. Grounding Head 통과 (Cross-Attention 수행)
        pred_bbox = self.grounding_head(image_feat, text_emb)
        
        return pred_bbox


def load_student_encoder(path, text_dim, backbone_name, device):
    """
    Student Encoder 로드
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
        try:
            ckpt = torch.load(path, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            
            # Strict=False로 로드 (Multi-Scale 구조 변경이나 Head 차이 대응)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            
            if missing:
                # print(f"[Info] Missing keys: {missing}")
                pass
        except Exception as e:
            print(f"[Error] Failed to load weights: {e}")
            print("[Model] Fallback to ImageNet pretrained weights.")
    else:
        if path:
            print(f"[Warning] Checkpoint not found at '{path}'.")
        print("[Model] Initializing with ImageNet pretrained weights (No Distillation).")
        
    return model.to(device)