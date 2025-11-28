import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from pathlib import Path
import numpy as np

# CLIP 표준 정규화 상수 (dataset.py와 동일)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711])

def denormalize_image(tensor):
    """
    [C, H, W] Tensor (Normalized) -> [H, W, C] Numpy (0~1)
    정규화된 텐서를 원래 이미지 색상으로 복구합니다.
    """
    # 1. Tensor -> Numpy & 차원 변경 ([C, H, W] -> [H, W, C])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    # 2. 역정규화 (Reverse Normalization): Original = (Norm * Std) + Mean
    # 브로드캐스팅을 위해 reshape가 필요할 수 있음 (H, W, 3) * (3,)
    img = (img * CLIP_STD) + CLIP_MEAN
    
    # 3. 값 범위를 0~1로 자르기 (Floating point error 방지)
    img = np.clip(img, 0, 1)
    return img

def save_visualization(image_tensor, command, pred_box, gt_box, save_path, iou_score):
    """이미지, 정답 박스(초록), 예측 박스(빨간), 명령어를 시각화하여 저장"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # 이미지 복구 및 그리기
    img_np = denormalize_image(image_tensor)
    h, w, _ = img_np.shape
    ax.imshow(img_np)
    
    # Ground Truth (Green)
    # box format: [x, y, w, h] normalized
    if gt_box is not None:
        gt_x = gt_box[0] * w
        gt_y = gt_box[1] * h
        gt_w = gt_box[2] * w
        gt_h = gt_box[3] * h
        rect_gt = patches.Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=3, edgecolor='lime', facecolor='none', label='Ground Truth')
        ax.add_patch(rect_gt)
    
    # Prediction (Red)
    if pred_box is not None:
        pred_x = pred_box[0] * w
        pred_y = pred_box[1] * h
        pred_w = pred_box[2] * w
        pred_h = pred_box[3] * h
        rect_pred = patches.Rectangle((pred_x, pred_y), pred_w, pred_h, linewidth=3, edgecolor='red', facecolor='none', label=f'Prediction (IoU: {iou_score:.2f})')
        ax.add_patch(rect_pred)
    
    # 텍스트
    wrapped_command = "\n".join(textwrap.wrap(command, width=60))
    plt.title(f"Command: {wrapped_command}", fontsize=15, pad=20)
    plt.legend(loc='upper right')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)