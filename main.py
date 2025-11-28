# main.py
import torch
import torch.optim as optim
import open_clip
import yaml
import os
import argparse
import sys
import datetime
from pathlib import Path

# 모듈 임포트
from data.download import download_and_setup_data
from models.student_model import load_student_encoder, Talk2CarModel
from utils.dataset import COCORegionTextDataset, collate_fn, get_clip_transform, get_augmented_transform
from utils.talk2car_dataset import Talk2CarDataset, talk2car_collate_fn
from utils.loss import DistillationLosses, Talk2CarLoss
from utils.training import (
    train_epoch, evaluate_retrieval, fine_tune_epoch, evaluate_talk2car, 
    inference_and_visualize, generate_predictions_json, adapt_teacher_to_talk2car
)
from utils.evaluation import detailed_talk2car_analysis

# === Logging Helper ===
class Logger(object):
    """콘솔 출력을 파일과 터미널에 동시 출력"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_config(config_path="config/default.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="MMDL Project: Enhanced Talk2Car Pipeline")
    parser.add_argument("--stage", type=str, default="all", 
                       choices=["0", "1", "2", "all", "test"], 
                       help="Execution stage (0=Teacher Adapt, 1=Distill, 2=Finetune, test=Inference)")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint for resuming")
    parser.add_argument("--visualize", action="store_true", 
                       help="Run visualization on test set")
    parser.add_argument("--generate_predictions", action="store_true",
                       help="Generate predictions.json for leaderboard submission")
    parser.add_argument("--detailed_analysis", action="store_true",
                       help="Run detailed performance analysis")
    return parser.parse_args()

def setup_experiment(cfg):
    """결과 디렉토리 생성 및 로깅 설정"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(cfg['ROOT_DIR']) / "results" / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 로깅
    log_file = result_dir / "training.log"
    sys.stdout = Logger(log_file)
    
    print(f"[Experiment] Result Directory: {result_dir}")
    print(f"[Experiment] Logs: {log_file}")

    # Config 백업
    with open(result_dir / "config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    return result_dir

def find_latest_checkpoint(root_dir, filename="distilled_weights.pth"):
    """최신 체크포인트 자동 탐색"""
    results_path = Path(root_dir) / "results"
    if not results_path.exists():
        return None
        
    result_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("result_")], 
                         key=lambda x: x.name, reverse=True)
    
    for d in result_dirs:
        ckpt_path = d / filename
        if ckpt_path.exists():
            print(f"[Auto-Resume] Found: {ckpt_path}")
            return ckpt_path
            
    return None

def main():
    # 1. 설정 및 환경
    args = parse_args()
    cfg = load_config()
    result_dir = setup_experiment(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"  🚀 MMDL Project - Enhanced Talk2Car Pipeline")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Mode: Stage {args.stage}")
    print(f"  Multi-Scale: {'Yes' if 'multiscale' in cfg['STUDENT_MODEL_BACKBONE'].lower() else 'No'}")
    print(f"  Augmentation: {'Albumentations' if cfg.get('AUGMENTATION', {}).get('USE_ALBUMENTATIONS') else 'Basic'}")
    print(f"{'='*70}\n")

    # 2. Teacher Model 로드
    print("[Init] Loading Teacher Model (OpenCLIP)...")
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        cfg['TEACHER_MODEL'], pretrained=cfg['TEACHER_PRETRAIN'], device=device
    )
    tokenizer = open_clip.get_tokenizer(cfg['TEACHER_MODEL'])
    
    # Teacher Freeze
    for p in teacher_model.parameters(): 
        p.requires_grad = False
    
    # 텍스트 임베딩 차원
    with torch.no_grad():
        text_dim = teacher_model.encode_text(tokenizer(["test"]).to(device)).shape[-1]

    # 3. Student Model Path 결정
    load_weights_path = None
    
    if args.resume:
        load_weights_path = Path(args.resume)
        if not load_weights_path.exists():
            print(f"[Error] Checkpoint not found: {load_weights_path}")
            sys.exit(1)
    elif cfg.get('STUDENT_WEIGHTS_PATH'):
        default_path = Path(cfg['ROOT_DIR']) / cfg['STUDENT_WEIGHTS_PATH']
        if default_path.exists():
            load_weights_path = default_path
    
    if args.stage in ["2", "test"] and not load_weights_path:
        load_weights_path = find_latest_checkpoint(cfg['ROOT_DIR'])
        if not load_weights_path:
            print("[Warning] No checkpoint found. Training from scratch.")

    # 4. Student Model 초기화
    student_encoder = load_student_encoder(
        str(load_weights_path) if load_weights_path else "", 
        text_dim, cfg['STUDENT_MODEL_BACKBONE'], device
    )
    
    # 저장 경로
    save_student_path = result_dir / "distilled_weights.pth"
    save_final_path = result_dir / "talk2car_final.pth"
    save_teacher_adapted_path = result_dir / "teacher_talk2car_adapted.pth"

    # ==========================================================================
    # STAGE 0: Teacher Domain Adaptation (NEW)
    # ==========================================================================
    if args.stage in ["0", "all"]:
        if cfg.get('TEACHER_ADAPTATION', {}).get('ENABLED', False):
            print("\n\n>>> STAGE 0: Teacher Domain Adaptation <<<")
            
            t2c_dir = Path(cfg['ROOT_DIR']) / cfg['TALK2CAR']['DIR_NAME']
            
            # Teacher 적응
            teacher_model = adapt_teacher_to_talk2car(
                teacher_model, tokenizer, t2c_dir, device, cfg
            )
            
            # 저장
            torch.save(teacher_model.state_dict(), save_teacher_adapted_path)
            print(f"[Stage 0] Adapted Teacher saved to {save_teacher_adapted_path}")
        else:
            print("[Stage 0] Skipped (TEACHER_ADAPTATION.ENABLED=false)")

    # ==========================================================================
    # STAGE 1: Knowledge Distillation (Alignment)
    # ==========================================================================
    if args.stage in ["1", "all"]:
        print("\n\n>>> STAGE 1: Knowledge Distillation <<<")
        
        # COCO 데이터 다운로드
        COCO_DIR = download_and_setup_data(cfg)
        
        if COCO_DIR:
            # Transform (Augmentation 적용)
            if cfg.get('AUGMENTATION', {}).get('USE_ALBUMENTATIONS', False):
                print("[Stage 1] Using Albumentations augmentation")
                train_transform = get_augmented_transform(cfg['IMAGE_SIZE'], cfg)
            else:
                print("[Stage 1] Using basic CLIP transform")
                train_transform = get_clip_transform(cfg['IMAGE_SIZE'])
            
            # Dataset & Loader
            train_dataset = COCORegionTextDataset(
                COCO_DIR, cfg, transform=train_transform, 
                max_images=cfg['MAX_IMAGES_TRAINING']
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=cfg['TRAIN']['BATCH_SIZE'], 
                shuffle=True, 
                collate_fn=collate_fn, 
                num_workers=cfg['TRAIN']['NUM_WORKERS']
            )
            
            # Loss & Optimizer
            loss_fn = DistillationLosses(temperature=cfg['TRAIN']['TEMPERATURE']).to(device)
            optimizer = optim.AdamW(
                list(student_encoder.parameters()) + list(loss_fn.parameters()),
                lr=float(cfg['TRAIN']['LEARNING_RATE']),
                weight_decay=float(cfg['TRAIN']['WEIGHT_DECAY'])
            )
            
            # Training Loop
            print(f"\n[Stage 1] Training for {cfg['TRAIN']['NUM_EPOCHS']} epochs...")
            for epoch in range(1, cfg['TRAIN']['NUM_EPOCHS'] + 1):
                train_epoch(student_encoder, loss_fn, optimizer, train_loader, 
                           tokenizer, teacher_model, device, cfg, epoch)
            
            # Save
            torch.save({'state_dict': student_encoder.state_dict()}, save_student_path)
            print(f"\n[Stage 1] ✅ Model saved to {save_student_path}")
            
            # Evaluate
            print("\n[Stage 1] Evaluating retrieval performance...")
            evaluate_retrieval(student_encoder, train_loader, tokenizer, teacher_model, device, cfg)
        else:
            print("[Stage 1] ❌ Error: Missing COCO data. Skipping Stage 1.")

    # ==========================================================================
    # STAGE 2: Talk2Car Fine-tuning (Grounding)
    # ==========================================================================
    if args.stage in ["2", "all"]:
        print("\n\n>>> STAGE 2: Talk2Car Fine-tuning <<<")
        
        # Stage 2 단독 실행 시 로드 확인
        if args.stage == "2":
            if load_weights_path and load_weights_path.exists():
                print(f"[Stage 2] Loading Stage 1 weights from {load_weights_path}")
            else:
                print(f"[Stage 2] ⚠️ No Stage 1 weights. Using ImageNet pretrained.")

        # 1. 통합 모델 초기화
        head_type = cfg['TALK2CAR']['HEAD_TYPE']
        talk2car_model = Talk2CarModel(student_encoder, text_dim, head_type=head_type).to(device)
        
        # 2. Dataset 준비
        t2c_dir = Path(cfg['ROOT_DIR']) / cfg['TALK2CAR']['DIR_NAME']
        
        # Train Dataset (Augmentation 적용)
        print("[Stage 2] Loading Train Set...")
        train_dataset = Talk2CarDataset(t2c_dir, cfg, split='train', transform=None)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], 
            shuffle=True,
            collate_fn=talk2car_collate_fn, 
            num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )

        # Validation Dataset
        print("[Stage 2] Loading Validation Set...")
        val_dataset = Talk2CarDataset(t2c_dir, cfg, split='val', transform=None)
        if len(val_dataset) == 0:
            print("[Warning] Val set empty. Using 'test' as validation...")
            val_dataset = Talk2CarDataset(t2c_dir, cfg, split='test', transform=None)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], 
            shuffle=False,
            collate_fn=talk2car_collate_fn, 
            num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )
        
        # 3. Loss & Optimizer
        t2c_loss_fn = Talk2CarLoss().to(device)
        ft_optimizer = optim.AdamW(
            talk2car_model.parameters(),
            lr=float(cfg['TALK2CAR']['FINE_TUNE']['LEARNING_RATE'])
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            ft_optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        best_iou = 0.0
        
        # 4. Training Loop
        print(f"\n[Stage 2] Training for {cfg['TALK2CAR']['FINE_TUNE']['NUM_EPOCHS']} epochs...")
        for epoch in range(1, cfg['TALK2CAR']['FINE_TUNE']['NUM_EPOCHS'] + 1):
            # Train
            fine_tune_epoch(talk2car_model, t2c_loss_fn, ft_optimizer, train_loader, 
                           tokenizer, teacher_model, device, epoch, cfg)
            
            # Evaluate
            avg_iou, ap50 = evaluate_talk2car(talk2car_model, val_loader, tokenizer, 
                                              teacher_model, device, cfg)
            
            # Scheduler
            scheduler.step(avg_iou)
            
            # Save Best
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(talk2car_model.state_dict(), save_final_path)
                print(f"[Stage 2] 🏆 New best model saved (IoU: {best_iou:.4f}) to {save_final_path}")

        print(f"\n[Stage 2] ✅ Training completed. Best IoU: {best_iou:.4f}")
        
        # Detailed Analysis
        if args.detailed_analysis:
            print("\n[Stage 2] Running detailed analysis...")
            talk2car_model.load_state_dict(torch.load(save_final_path))
            detailed_talk2car_analysis(talk2car_model, val_loader, tokenizer, teacher_model, device)

    # ==========================================================================
    # TEST INFERENCE MODE (NEW)
    # ==========================================================================
    if args.stage == "test" or args.generate_predictions:
        print("\n\n>>> TEST INFERENCE MODE <<<")
        
        # 1. Best Model 로드
        best_model_path = save_final_path if save_final_path.exists() else None
        
        if not best_model_path:
            best_model_path = find_latest_checkpoint(cfg['ROOT_DIR'], "talk2car_final.pth")
        
        if not best_model_path or not best_model_path.exists():
            print("[Error] ❌ No trained model found. Please train Stage 2 first.")
            sys.exit(1)
        
        print(f"[Test] Loading model from {best_model_path}")
        
        # 2. 모델 초기화
        # Student Encoder 로드
        student_weights_path = str(best_model_path).replace("talk2car_final.pth", "distilled_weights.pth")
        if not Path(student_weights_path).exists():
            student_weights_path = find_latest_checkpoint(cfg['ROOT_DIR'], "distilled_weights.pth")
        
        student_encoder = load_student_encoder(
            str(student_weights_path) if student_weights_path else "",
            text_dim, cfg['STUDENT_MODEL_BACKBONE'], device
        )
        
        talk2car_model = Talk2CarModel(
            student_encoder, text_dim, 
            head_type=cfg['TALK2CAR']['HEAD_TYPE']
        ).to(device)
        
        talk2car_model.load_state_dict(torch.load(best_model_path))
        talk2car_model.eval()
        
        # 3. Test Dataset 로드
        t2c_dir = Path(cfg['ROOT_DIR']) / cfg['TALK2CAR']['DIR_NAME']
        
        test_dataset = Talk2CarDataset(t2c_dir, cfg, split='test', transform=None)
        
        if len(test_dataset) == 0:
            print("[Warning] Test set empty. Using 'val' set for inference.")
            test_dataset = val_dataset
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], 
            shuffle=False,
            collate_fn=talk2car_collate_fn, 
            num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )
        
        # 4. predictions.json 생성
        if args.generate_predictions or args.stage == "test":
            predictions_path = result_dir / "predictions.json"
            generate_predictions_json(
                talk2car_model, test_loader, tokenizer, teacher_model, 
                device, predictions_path, cfg
            )
        
        # 5. Visualization (선택적)
        if args.visualize:
            print("\n[Test] Generating visualizations...")
            vis_save_dir = result_dir / "visualizations"
            inference_and_visualize(
                talk2car_model, test_loader, tokenizer, teacher_model, device, 
                save_dir=vis_save_dir, 
                max_vis=cfg['TALK2CAR'].get('VIS_COUNT', 50)
            )

    print(f"\n{'='*70}")
    print(f"  ✅ Execution Completed Successfully!")
    if 'best_iou' in locals():
        print(f"  📊 Final Best IoU: {best_iou:.4f}")
    print(f"  📁 Results saved to: {result_dir}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
