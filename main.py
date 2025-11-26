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
from utils.loss import DistillationLosses
from utils.training import train_epoch, evaluate_retrieval, fine_tune_epoch, evaluate_talk2car

# === Logging Helper Class ===
class Logger(object):
    """콘솔 출력(stdout/stderr)을 파일과 터미널 양쪽에 동시에 출력하는 헬퍼 클래스"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 즉시 파일에 쓰기

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_config(config_path="config/default.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="MMDL Project: Distillation & Fine-tuning")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"])
    return parser.parse_args()

def setup_experiment(cfg):
    """결과 디렉토리 생성 및 로깅/Config 설정"""
    # 1. 결과 디렉토리 생성 (results/result_YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(cfg['ROOT_DIR']) / "results" / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 로깅 설정 (stdout과 stderr를 가로채서 파일에도 기록)
    log_file = result_dir / "training.log"
    sys.stdout = Logger(log_file)
    # 에러 메시지나 tqdm 출력도 로그에 남기기 위해 stderr도 리다이렉트 (선택사항)
    # sys.stderr = Logger(log_file) 
    
    print(f"[Experiment] Result Directory Created: {result_dir}")
    print(f"[Experiment] Logs will be saved to: {log_file}")

    # 3. Config 백업
    with open(result_dir / "config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Experiment] Config saved to: {result_dir / 'config.yaml'}")
    
    return result_dir

def main():
    args = parse_args()
    cfg = load_config()
    
    # 실험 환경 셋업 (폴더 생성, 로그 시작)
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

    # Student Model Path (Load용)
    # 기존에 학습된 가중치가 있다면 로드할 경로 (Config 기준)
    load_weights_path = Path(cfg['ROOT_DIR']) / cfg['STUDENT_WEIGHTS_PATH']
    
    # 모델 초기화
    student_encoder = load_student_encoder(
        load_weights_path if load_weights_path.exists() else "", 
        text_dim, cfg['STUDENT_MODEL_BACKBONE'], device
    )

    # 저장할 경로 정의 (Result Dir 내부)
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
            
            # Stage 1 결과 저장 (Result Dir에)
            torch.save({'state_dict': student_encoder.state_dict()}, save_student_path)
            print(f"[Stage 1] Model saved to {save_student_path}")
            
            evaluate_retrieval(student_encoder, train_loader, tokenizer, teacher_model, device, cfg)
        else:
            print("[Stage 1] Error: Missing COCO data. Skipping Stage 1.")

    # STAGE 2
    if args.stage in ["2", "all"]:
        print("\n\n>>> STARTING STAGE 2: Talk2Car Fine-tuning <<<")
        if args.stage == "2":
            # Stage 2 단독 실행 시에는 기존 Config 경로 또는 Result Dir 경로 확인 필요
            # 여기서는 Config에 지정된 경로를 우선으로 로드 시도
            if not load_weights_path.exists():
                print(f"[Warning] Stage 1 weights not found at {load_weights_path}. Using ImageNet pretrained.")
            else:
                print(f"[Init] Reloading Stage 1 weights from {load_weights_path}...")
                student_encoder = load_student_encoder(load_weights_path, text_dim, cfg['STUDENT_MODEL_BACKBONE'], device)
        
        # 만약 Stage 1->2 연속 실행('all')이라면, 메모리에 있는 student_encoder가 그대로 사용됨 (가장 최신 상태)

        # 1. 2단계 모델 초기화 (Head Type 설정 반영)
        head_type = cfg['TALK2CAR']['HEAD_TYPE']
        talk2car_model = Talk2CarModel(student_encoder, text_dim, head_type=head_type).to(device)
        
        t2c_dir = Path(cfg['ROOT_DIR']) / cfg['TALK2CAR']['DIR_NAME']
        clip_transform = get_clip_transform(cfg['IMAGE_SIZE'])
        
        t2c_dataset = Talk2CarDataset(t2c_dir, cfg, transform=clip_transform)
        t2c_loader = torch.utils.data.DataLoader(
            t2c_dataset, batch_size=cfg['TALK2CAR']['FINE_TUNE']['BATCH_SIZE'], shuffle=True,
            collate_fn=talk2car_collate_fn, num_workers=cfg['TALK2CAR']['FINE_TUNE']['NUM_WORKERS']
        )
        
        ft_optimizer = optim.AdamW(
            talk2car_model.parameters(),
            lr=cfg['TALK2CAR']['FINE_TUNE']['LEARNING_RATE']
        )
        
        for epoch in range(1, cfg['TALK2CAR']['FINE_TUNE']['NUM_EPOCHS'] + 1):
            fine_tune_epoch(talk2car_model, ft_optimizer, t2c_loader, tokenizer, teacher_model, device, epoch, cfg)
            evaluate_talk2car(talk2car_model, t2c_loader, tokenizer, teacher_model, device, cfg)

        # 최종 모델 저장 (Result Dir에)
        torch.save(talk2car_model.state_dict(), save_final_path)
        print(f"\n[Stage 2] Final Talk2Car model saved to {save_final_path}")

    print("=== Execution Completed ===")

if __name__ == '__main__':
    main()