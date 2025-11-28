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
from utils.dataset import COCORegionTextDataset, collate_fn, get_clip_transform
from utils.talk2car_dataset import Talk2CarDataset, talk2car_collate_fn
from utils.loss import DistillationLosses, Talk2CarLoss
from utils.training import train_epoch, evaluate_retrieval, fine_tune_epoch, evaluate_talk2car, inference_and_visualize

# === Logging Helper Class ===
class Logger(object):
    """콘솔 출력(stdout)을 파일과 터미널 양쪽에 동시에 출력하는 헬퍼 클래스"""
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
    parser = argparse.ArgumentParser(description="MMDL Project: Distillation & Fine-tuning")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"], help="Execution stage")
    parser.add_argument("--resume", type=str, default=None, help="Path to stage 1 checkpoint for stage 2 fine-tuning")
    parser.add_argument("--visualize", action="store_true", help="Run visualization on test set after training")
    return parser.parse_args()

def setup_experiment(cfg):
    """결과 디렉토리 생성 및 로깅/Config 설정"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(cfg['ROOT_DIR']) / "results" / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 로깅 설정 (stdout을 가로채서 파일에도 기록)
    log_file = result_dir / "training.log"
    sys.stdout = Logger(log_file)
    
    print(f"[Experiment] Result Directory Created: {result_dir}")
    print(f"[Experiment] Logs will be saved to: {log_file}")

    # Config 백업
    with open(result_dir / "config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    return result_dir

def find_latest_checkpoint(root_dir, filename="distilled_weights.pth"):
    """results 폴더 내에서 가장 최근에 생성된 가중치 파일을 찾습니다."""
    results_path = Path(root_dir) / "results"
    if not results_path.exists():
        return None
        
    # result_YYYYMMDD_HHMMSS 형식의 폴더들을 찾아서 정렬 (최신순)
    result_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("result_")], 
                         key=lambda x: x.name, reverse=True)
    
    for d in result_dirs:
        ckpt_path = d / filename
        if ckpt_path.exists():
            print(f"[Auto-Resume] Found latest checkpoint: {ckpt_path}")
            return ckpt_path
            
    return None

def main():
    # 1. 설정 및 실험 환경 준비
    args = parse_args()
    cfg = load_config()
    result_dir = setup_experiment(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== Project MMDL Started on {device} (Mode: Stage {args.stage}) ===")

    # 2. Teacher Model 로드 (OpenCLIP) - Frozen 상태
    print("\n[Init] Loading Teacher Model (OpenCLIP)...")
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        cfg['TEACHER_MODEL'], pretrained=cfg['TEACHER_PRETRAIN'], device=device
    )
    tokenizer = open_clip.get_tokenizer(cfg['TEACHER_MODEL'])
    for p in teacher_model.parameters(): p.requires_grad = False
    
    # 텍스트 임베딩 차원 확인
    with torch.no_grad():
        text_dim = teacher_model.encode_text(tokenizer(["test"]).to(device)).shape[-1]

    # 3. Student Model Path 결정 (Load용)
    load_weights_path = None
    
    # 1순위: 사용자가 --resume으로 지정한 경로
    if args.resume:
        load_weights_path = Path(args.resume)
        if not load_weights_path.exists():
            print(f"[Error] Specified checkpoint not found: {load_weights_path}")
            sys.exit(1)
            
    # 2순위: Config에 지정된 기본 경로
    elif cfg.get('STUDENT_WEIGHTS_PATH'):
        default_path = Path(cfg['ROOT_DIR']) / cfg['STUDENT_WEIGHTS_PATH']
        if default_path.exists():
            load_weights_path = default_path

    # 3순위: (Stage 2 단독 실행 시) 가장 최근 결과 자동 탐색
    if args.stage == "2" and not load_weights_path:
        load_weights_path = find_latest_checkpoint(cfg['ROOT_DIR'])
        if not load_weights_path:
            print("[Warning] No checkpoint found. Training from scratch (ImageNet init).")

    # 4. Student Model (Image Encoder) 초기화
    student_encoder = load_student_encoder(
        load_weights_path if load_weights_path else "", 
        text_dim, cfg['STUDENT_MODEL_BACKBONE'], device
    )
    
    # 이번 실행 결과를 저장할 경로 정의
    save_student_path = result_dir / "distilled_weights.pth"
    save_final_path = result_dir / "talk2car_final.pth"

    # ==========================================================================
    # STAGE 1: Knowledge Distillation (Alignment)
    # ==========================================================================
    if args.stage in ["1", "all"]:
        print("\n\n>>> STARTING STAGE 1: Knowledge Distillation <<<")
        
        # 데이터 다운로드 및 준비
        COCO_DIR = download_and_setup_data(cfg)
        
        if COCO_DIR:
            # Dataset & Loader
            clip_transform = get_clip_transform(cfg['IMAGE_SIZE'])
            train_dataset = COCORegionTextDataset(COCO_DIR, cfg, transform=clip_transform, max_images=cfg['MAX_IMAGES_TRAINING'])
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, 
                collate_fn=collate_fn, num_workers=cfg['TRAIN']['NUM_WORKERS']
            )
            
            # Loss & Optimizer
            loss_fn = DistillationLosses(temperature=cfg['TRAIN']['TEMPERATURE']).to(device)
            optimizer = optim.AdamW(
                list(student_encoder.parameters()) + list(loss_fn.parameters()),
                lr=cfg['TRAIN']['LEARNING_RATE'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY']
            )
            
            # Training Loop
            for epoch in range(1, cfg['TRAIN']['NUM_EPOCHS'] + 1):
                train_epoch(student_encoder, loss_fn, optimizer, train_loader, tokenizer, teacher_model, device, cfg, epoch)
            
            # Save & Evaluate
            torch.save({'state_dict': student_encoder.state_dict()}, save_student_path)
            print(f"[Stage 1] Model saved to {save_student_path}")
            
            evaluate_retrieval(student_encoder, train_loader, tokenizer, teacher_model, device, cfg)
        else:
            print("[Stage 1] Error: Missing COCO data. Skipping Stage 1.")

    # ==========================================================================
    # STAGE 2: Talk2Car Fine-tuning (Grounding)
    # ==========================================================================
    if args.stage in ["2", "all"]:
        print("\n\n>>> STARTING STAGE 2: Talk2Car Fine-tuning <<<")
        
        # Stage 2 단독 실행일 때 로드 메시지 출력
        if args.stage == "2":
            if load_weights_path and load_weights_path.exists():
                print(f"[Init] Reloading Stage 1 weights from {load_weights_path}...")
                # 이미 위에서 초기화했지만, 명시적으로 확인 (student_encoder 객체는 유지됨)
            else:
                print(f"[Warning] Using ImageNet pretrained backbone (No Distillation).")

        # 1. 2단계 전체 모델 초기화 (Encoder + Grounding Head)
        head_type = cfg['TALK2CAR']['HEAD_TYPE']
        talk2car_model = Talk2CarModel(student_encoder, text_dim, head_type=head_type).to(device)
        
        # 2. 데이터셋 준비 (Train / Val Split)
        t2c_dir = Path(cfg['ROOT_DIR']) / cfg['TALK2CAR']['DIR_NAME']
        clip_transform = get_clip_transform(cfg['IMAGE_SIZE'])
        
        # Train Dataset
        print("[Dataset] Loading Train Set...")
        train_dataset = Talk2CarDataset(t2c_dir, cfg, split='train', transform=clip_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], shuffle=True,
            collate_fn=talk2car_collate_fn, num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )

        # Validation Dataset
        print("[Dataset] Loading Validation Set...")
        val_dataset = Talk2CarDataset(t2c_dir, cfg, split='val', transform=clip_transform)
        # Validation 데이터가 없으면 Test 데이터로 대체 시도
        if len(val_dataset) == 0:
             print("[Warning] Val set empty. Trying 'test' split as validation...")
             val_dataset = Talk2CarDataset(t2c_dir, cfg, split='test', transform=clip_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], shuffle=False,
            collate_fn=talk2car_collate_fn, num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )
        
        # 3. Loss & Optimizer & Scheduler
        t2c_loss_fn = Talk2CarLoss().to(device)
        ft_optimizer = optim.AdamW(
            talk2car_model.parameters(),
            lr=cfg['TALK2CAR']['FINE_TUNE']['LEARNING_RATE']
        )
        
        # ReduceLROnPlateau: 성능(IoU)이 정체되면 LR을 감소시킴
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            ft_optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        best_iou = 0.0
        
        # 4. Training Loop
        for epoch in range(1, cfg['TALK2CAR']['FINE_TUNE']['NUM_EPOCHS'] + 1):
            # Train
            fine_tune_epoch(talk2car_model, t2c_loss_fn, ft_optimizer, train_loader, tokenizer, teacher_model, device, epoch, cfg)
            
            # Evaluate (on Validation Set)
            avg_iou, ap50 = evaluate_talk2car(talk2car_model, val_loader, tokenizer, teacher_model, device, cfg)
            
            # Scheduler Step
            scheduler.step(avg_iou)
            
            # Save Best Model
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(talk2car_model.state_dict(), save_final_path)
                print(f"[Stage 2] New best model saved (IoU: {best_iou:.4f}) to {save_final_path}")

        # ==========================================================================
        # VISUALIZATION ON TEST SET (Optional)
        # ==========================================================================
        if args.visualize:
            print("\n\n>>> RUNNING VISUALIZATION ON TEST SET <<<")
            
            # Best Model 로드
            if save_final_path.exists():
                print(f"[Vis] Loading best model from {save_final_path}")
                talk2car_model.load_state_dict(torch.load(save_final_path))
            
            # Test Set 로드
            test_dataset = Talk2CarDataset(t2c_dir, cfg, split='test', transform=clip_transform)
            if len(test_dataset) == 0:
                print("[Warning] Test set empty. Using 'val' set for visualization.")
                test_dataset = val_dataset
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], shuffle=False,
                collate_fn=talk2car_collate_fn, num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
            )
            
            vis_save_dir = result_dir / "visualizations"
            inference_and_visualize(
                talk2car_model, test_loader, tokenizer, teacher_model, device, 
                save_dir=vis_save_dir, 
                max_vis=cfg['TALK2CAR'].get('VIS_COUNT', 50)
            )

    print(f"=== Execution Completed. Final Best IoU: {best_iou if 'best_iou' in locals() else 0:.4f} ===")

if __name__ == '__main__':
    main()