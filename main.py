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
from utils.training import train_epoch, evaluate_retrieval, fine_tune_epoch, evaluate_talk2car

# === Logging Helper Class ===
class Logger(object):
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
    return parser.parse_args()

def setup_experiment(cfg):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(cfg['ROOT_DIR']) / "results" / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = result_dir / "training.log"
    sys.stdout = Logger(log_file)
    
    print(f"[Experiment] Result Directory Created: {result_dir}")
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
    args = parse_args()
    cfg = load_config()
    result_dir = setup_experiment(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== Project MMDL Started on {device} (Mode: Stage {args.stage}) ===")

    # Teacher Model
    print("\n[Init] Loading Teacher Model (OpenCLIP)...")
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        cfg['TEACHER_MODEL'], pretrained=cfg['TEACHER_PRETRAIN'], device=device
    )
    tokenizer = open_clip.get_tokenizer(cfg['TEACHER_MODEL'])
    for p in teacher_model.parameters(): p.requires_grad = False
    
    with torch.no_grad():
        text_dim = teacher_model.encode_text(tokenizer(["test"]).to(device)).shape[-1]

    # Student Model Path 설정 (Load용)
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

    # 모델 초기화 (가중치 로드 시도)
    student_encoder = load_student_encoder(
        load_weights_path if load_weights_path else "", 
        text_dim, cfg['STUDENT_MODEL_BACKBONE'], device
    )
    
    # 저장 경로 (이번 실행 결과)
    save_student_path = result_dir / "distilled_weights.pth"
    save_final_path = result_dir / "talk2car_final.pth"

    # STAGE 1
    if args.stage in ["1", "all"]:
        print("\n\n>>> STARTING STAGE 1: Knowledge Distillation <<<")
        COCO_DIR = download_and_setup_data(cfg)
        if COCO_DIR:
            clip_transform = get_clip_transform(cfg['IMAGE_SIZE'])
            train_dataset = COCORegionTextDataset(COCO_DIR, cfg, transform=clip_transform, max_images=cfg['MAX_IMAGES_TRAINING'])
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, 
                collate_fn=collate_fn, num_workers=cfg['TRAIN']['NUM_WORKERS']
            )
            
            loss_fn = DistillationLosses(temperature=cfg['TRAIN']['TEMPERATURE']).to(device)
            optimizer = optim.AdamW(
                list(student_encoder.parameters()) + list(loss_fn.parameters()),
                lr=cfg['TRAIN']['LEARNING_RATE'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY']
            )
            
            for epoch in range(1, cfg['TRAIN']['NUM_EPOCHS'] + 1):
                train_epoch(student_encoder, loss_fn, optimizer, train_loader, tokenizer, teacher_model, device, cfg, epoch)
            
            torch.save({'state_dict': student_encoder.state_dict()}, save_student_path)
            print(f"[Stage 1] Model saved to {save_student_path}")
            evaluate_retrieval(student_encoder, train_loader, tokenizer, teacher_model, device, cfg)
        else:
            print("[Stage 1] Error: Missing COCO data. Skipping Stage 1.")

    # STAGE 2
    if args.stage in ["2", "all"]:
        print("\n\n>>> STARTING STAGE 2: Talk2Car Fine-tuning <<<")
        # Note: 'all' 모드일 경우 student_encoder는 이미 학습된 상태 그대로 넘어옴
        # '2' 모드일 경우 위에서 로드한 상태로 시작됨

        head_type = cfg['TALK2CAR']['HEAD_TYPE']
        talk2car_model = Talk2CarModel(student_encoder, text_dim, head_type=head_type).to(device)
        
        t2c_dir = Path(cfg['ROOT_DIR']) / cfg['TALK2CAR']['DIR_NAME']
        clip_transform = get_clip_transform(cfg['IMAGE_SIZE'])
        
        t2c_dataset = Talk2CarDataset(t2c_dir, cfg, transform=clip_transform)
        t2c_loader = torch.utils.data.DataLoader(
            t2c_dataset, batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], shuffle=True,
            collate_fn=talk2car_collate_fn, num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )
        
        t2c_loss_fn = Talk2CarLoss().to(device)
        ft_optimizer = optim.AdamW(
            talk2car_model.parameters(),
            lr=cfg['TALK2CAR']['FINE_TUNE']['LEARNING_RATE']
        )
        
        for epoch in range(1, cfg['TALK2CAR']['FINE_TUNE']['NUM_EPOCHS'] + 1):
            fine_tune_epoch(talk2car_model, t2c_loss_fn, ft_optimizer, t2c_loader, tokenizer, teacher_model, device, epoch, cfg)
            evaluate_talk2car(talk2car_model, t2c_loader, tokenizer, teacher_model, device, cfg)

        torch.save(talk2car_model.state_dict(), save_final_path)
        print(f"\n[Stage 2] Final Talk2Car model saved to {save_final_path}")

    print("=== Execution Completed ===")

if __name__ == '__main__':
    main()