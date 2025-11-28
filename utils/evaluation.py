# utils/evaluation.py
"""
팀원 코드 스타일의 상세 성능 분석
- 명령어 길이별 성능
- 객체 크기별 성능
- 예측 분포 분석
"""
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

def detailed_talk2car_analysis(model, loader, tokenizer, teacher_model, device):
    """
    Talk2Car 성능을 다각도로 분석
    """
    from utils.training import encode_text
    
    model.eval()
    
    results = {
        'by_command_length': {},  # 명령어 길이별
        'by_object_size': {},     # 객체 크기별
        'prediction_dist': Counter(),
        'all_ious': []
    }
    
    print("\n=== Detailed Talk2Car Analysis ===")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing"):
            images, commands, gt_bboxes, _ = batch
            images = images.to(device)
            gt_bboxes = gt_bboxes.to(device)
            
            text_emb = encode_text(commands, tokenizer, teacher_model, device)
            pred_bboxes = model(images, text_emb)
            
            # IoU 계산
            pred_x1 = pred_bboxes[:, 0]
            pred_y1 = pred_bboxes[:, 1]
            pred_x2 = pred_bboxes[:, 0] + pred_bboxes[:, 2]
            pred_y2 = pred_bboxes[:, 1] + pred_bboxes[:, 3]
            
            gt_x1 = gt_bboxes[:, 0]
            gt_y1 = gt_bboxes[:, 1]
            gt_x2 = gt_bboxes[:, 0] + gt_bboxes[:, 2]
            gt_y2 = gt_bboxes[:, 1] + gt_bboxes[:, 3]

            inter_x1 = torch.max(pred_x1, gt_x1)
            inter_y1 = torch.max(pred_y1, gt_y1)
            inter_x2 = torch.min(pred_x2, gt_x2)
            inter_y2 = torch.min(pred_y2, gt_y2)
            
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            
            pred_area = pred_bboxes[:, 2] * pred_bboxes[:, 3]
            gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            union_area = pred_area + gt_area - inter_area
            
            ious = inter_area / (union_area + 1e-6)
            
            # 분석
            for i, (cmd, iou, bbox) in enumerate(zip(commands, ious, gt_bboxes)):
                iou_val = iou.item()
                results['all_ious'].append(iou_val)
                
                # 1. 명령어 길이별
                word_count = len(cmd.split())
                length_bin = f"{(word_count // 3) * 3}-{(word_count // 3) * 3 + 2} words"
                
                if length_bin not in results['by_command_length']:
                    results['by_command_length'][length_bin] = []
                results['by_command_length'][length_bin].append(iou_val)
                
                # 2. 객체 크기별
                area = bbox[2].item() * bbox[3].item()
                if area < 0.05:
                    size_bin = 'very_small'
                elif area < 0.15:
                    size_bin = 'small'
                elif area < 0.35:
                    size_bin = 'medium'
                else:
                    size_bin = 'large'
                
                if size_bin not in results['by_object_size']:
                    results['by_object_size'][size_bin] = []
                results['by_object_size'][size_bin].append(iou_val)
    
    # 결과 출력
    print("\n[By Command Length]")
    print(f"{'Length':<15} | {'Count':<8} | {'Avg IoU':<10} | {'AP50':<10}")
    print("-" * 50)
    for length in sorted(results['by_command_length'].keys()):
        ious = results['by_command_length'][length]
        avg_iou = np.mean(ious)
        ap50 = sum([1 for iou in ious if iou >= 0.5]) / len(ious) * 100
        print(f"{length:<15} | {len(ious):<8} | {avg_iou:<10.3f} | {ap50:<10.1f}%")
    
    print("\n[By Object Size]")
    print(f"{'Size':<15} | {'Count':<8} | {'Avg IoU':<10} | {'AP50':<10}")
    print("-" * 50)
    size_order = ['very_small', 'small', 'medium', 'large']
    for size in size_order:
        if size in results['by_object_size']:
            ious = results['by_object_size'][size]
            avg_iou = np.mean(ious)
            ap50 = sum([1 for iou in ious if iou >= 0.5]) / len(ious) * 100
            print(f"{size:<15} | {len(ious):<8} | {avg_iou:<10.3f} | {ap50:<10.1f}%")
    
    print(f"\n[Overall Statistics]")
    print(f"  Total Samples: {len(results['all_ious'])}")
    print(f"  Mean IoU: {np.mean(results['all_ious']):.4f}")
    print(f"  Median IoU: {np.median(results['all_ious']):.4f}")
    print(f"  AP50: {sum([1 for iou in results['all_ious'] if iou >= 0.5]) / len(results['all_ious']) * 100:.2f}%")
    print(f"  AP75: {sum([1 for iou in results['all_ious'] if iou >= 0.75]) / len(results['all_ious']) * 100:.2f}%")
    
    return results
