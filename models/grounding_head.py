import torch
import torch.nn as nn

class ProjectionGroundingHead(nn.Module):
    """
    Simple MLP-based Grounding Head (Baseline).
    Concat(Image, Text) -> MLP -> BBox
    """
    def __init__(self, image_feat_dim, text_dim, output_dim=4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, image_feat_dim)
        
        self.combiner = nn.Sequential(
            nn.Linear(image_feat_dim * 2, image_feat_dim),
            nn.BatchNorm1d(image_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(image_feat_dim, image_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(image_feat_dim // 2, output_dim),
            nn.Sigmoid() 
        )
        
    def forward(self, image_feat, text_emb):
        # Projection Head는 [B, C] 형태의 Global Feature를 기대함
        # 만약 입력이 [B, C, H, W]라면 Global Pooling 수행
        if image_feat.dim() == 4:
            image_feat = image_feat.mean(dim=[2, 3]) # GAP
            
        text_projected = self.text_proj(text_emb)
        combined_feat = torch.cat([image_feat, text_projected], dim=-1)
        prediction = self.combiner(combined_feat)
        return prediction

class TransformerGroundingHead(nn.Module):
    """
    Transformer Decoder-based Grounding Head (SOTA Approach).
    Cross-Attend(Query=Text, Key/Value=Image) -> BBox
    """
    def __init__(self, image_feat_dim, text_dim, hidden_dim=256, nhead=4, num_layers=3, output_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Conv2d(image_feat_dim, hidden_dim, kernel_size=1)
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=1024, dropout=0.1, activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def get_positional_encoding(self, batch_size, h, w, device):
        x_emb = self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed[:h].unsqueeze(1).repeat(1, w, 1)
        pos = torch.cat([x_emb, y_emb], dim=-1).flatten(0, 1).unsqueeze(1).repeat(1, batch_size, 1)
        return pos.to(device)

    def forward(self, image_feat, text_emb):
        # Transformer Head는 [B, C, H, W] 형태의 Spatial Feature를 기대함
        if image_feat.dim() == 2:
            raise ValueError("TransformerHead requires spatial features [B, C, H, W], but got global features [B, C].")

        bs, c, h, w = image_feat.shape
        src = self.input_proj(image_feat).flatten(2).permute(2, 0, 1)
        pos = self.get_positional_encoding(bs, h, w, src.device)
        tgt = self.query_proj(text_emb).unsqueeze(0)
        
        hs = self.decoder(tgt, src + pos)
        output = hs.squeeze(0)
        prediction = self.bbox_embed(output)
        return prediction