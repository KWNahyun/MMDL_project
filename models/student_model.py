import torch
import torch.nn as nn
import timm
import os
from models.grounding_head import ProjectionGroundingHead, TransformerGroundingHead

class DistilledConvNeXtTiny(nn.Module):
    def __init__(self, text_dim, backbone_name="convnext_tiny"):
        super().__init__()
        # Spatial Feature를 얻기 위해 global_pool='' 설정
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='')
        
        # Distillation(Stage 1)에서는 Global Feature가 필요하므로 별도 풀링 및 헤드 정의
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.num_features, text_dim)

    def forward(self, x):
        feat_map = self.backbone(x) # [B, C, H, W]
        pooled = self.pool(feat_map).flatten(1) # [B, C]
        emb  = self.head(pooled)
        return emb

class Talk2CarModel(nn.Module):
    """
    Stage 2 파인튜닝 모델. Config에 따라 Head Type을 선택합니다.
    """
    def __init__(self, distilled_encoder, text_dim, head_type):
        super().__init__()
        
        self.image_encoder = distilled_encoder
        encoder_feat_dim = distilled_encoder.backbone.num_features
        
        self.head_type = head_type
        print(f"[Model] Initializing Talk2CarModel with Head Type: {head_type}")

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
        # 1. 이미지 백본 통과 (항상 Spatial Feature [B, C, H, W] 반환)
        image_feat = self.image_encoder.backbone(images)
        
        # 2. 선택된 Grounding Head 통과
        # (ProjectionHead는 내부적으로 Pooling을 수행하고, Transformer는 Spatial Feature를 그대로 사용)
        pred_bbox = self.grounding_head(image_feat, text_emb)
        
        return pred_bbox

def load_student_encoder(path, text_dim, backbone_name, device):
    model = DistilledConvNeXtTiny(text_dim=text_dim, backbone_name=backbone_name)
    
    if path and os.path.exists(path):
        print(f"Loading student weights from: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
    else:
        print(f"Student weights not found at '{path}'. Initializing with ImageNet pretrained backbone.")
        
    return model.to(device)