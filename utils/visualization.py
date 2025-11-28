import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from pathlib import Path
import numpy as np

def denormalize_image(tensor):
    """
    [C, H, W] Tensor (0~1) -> [H, W, C] Numpy (0~1) for plotting
    """
    img = tensor.permute(1, 2, 0).cpu().numpy()
    # 값이 범위를 벗어나지 않게 클리핑
    img = np.clip(img, 0, 1)
    return img

def save_visualization(image_tensor, command, pred_box, gt_box, save_path, iou_score):
    """
    이미지와 BBox를 시각화하여 저장합니다.
    - Green: Ground Truth
    - Red: Prediction
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # 1. 이미지 그리기
    img_np = denormalize_image(image_tensor)
    h, w, _ = img_np.shape
    ax.imshow(img_np)
    
    # 2. BBox 그리기 (Format: x, y, w, h normalized)
    # Ground Truth (Green)
    gt_x = gt_box[0] * w
    gt_y = gt_box[1] * h
    gt_w = gt_box[2] * w
    gt_h = gt_box[3] * h
    
    rect_gt = patches.Rectangle(
        (gt_x, gt_y), gt_w, gt_h, 
        linewidth=3, edgecolor='lime', facecolor='none', label='Ground Truth'
    )
    ax.add_patch(rect_gt)
    
    # Prediction (Red)
    pred_x = pred_box[0] * w
    pred_y = pred_box[1] * h
    pred_w = pred_box[2] * w
    pred_h = pred_box[3] * h
    
    rect_pred = patches.Rectangle(
        (pred_x, pred_y), pred_w, pred_h, 
        linewidth=3, edgecolor='red', facecolor='none', label=f'Prediction (IoU: {iou_score:.2f})'
    )
    ax.add_patch(rect_pred)
    
    # 3. 텍스트 명령 및 범례 추가
    # 텍스트가 길면 줄바꿈
    wrapped_command = "\n".join(textwrap.wrap(command, width=60))
    plt.title(f"Command: {wrapped_command}", fontsize=15, pad=20)
    plt.legend(loc='upper right')
    plt.axis('off')
    
    # 4. 저장
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # 메모리 해제