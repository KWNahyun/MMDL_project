import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open_clip
import numpy as np
import time
import argparse
from tqdm import tqdm
from thop import profile
import yaml
import os
import glob
import urllib.request
import zipfile

# --- 프로젝트 모듈 임포트 ---
# MMDL 폴더에서 실행한다고 가정
from models.student_model import DistilledConvNeXtTiny, DistilledConvNeXtTinyMultiScale
from utils.dataset import COCORegionTextDataset, collate_fn, get_clip_transform

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Auto Evaluation with Auto-Download")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment folder (e.g., results/exp_01)")
    # [수정] 이미지상의 구조(MMDL/coco2017)에 맞춰 기본값을 현재 디렉토리('.')로 설정
    parser.add_argument("--data_root", type=str, default=".", help="Root directory containing coco2017 folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

# ==========================================
# 1. 데이터셋 자동 다운로드 및 준비 함수
# ==========================================
def download_and_extract(url, dest_dir, desc):
    """URL에서 파일을 다운로드하고 압축을 해제합니다."""
    filename = url.split('/')[-1]
    file_path = os.path.join(dest_dir, filename)
    
    # 1. 다운로드
    if not os.path.exists(file_path):
        print(f"\n[Download] Downloading {desc} ({filename})...")
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)
        except Exception as e:
            print(f"[Error] Download failed: {e}")
            return False
    else:
        print(f"[Check] {filename} found. Skipping download.")

    # 2. 압축 해제
    print(f"[Extract] Extracting {filename} to {dest_dir}...")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"[Done] Extraction complete.")
        
        # (선택) 압축 파일 삭제 (공간 확보)
        # os.remove(file_path) 
        return True
    except zipfile.BadZipFile:
        print(f"[Error] Bad zip file: {file_path}. Please delete it and try again.")
        return False

def prepare_coco_val(root_dir, coco_dir_name='coco2017'):
    """
    COCO val2017 이미지와 어노테이션 확인 후 없으면 다운로드
    """
    coco_path = os.path.join(root_dir, coco_dir_name)
    val_img_dir = os.path.join(coco_path, "val2017")
    ann_dir = os.path.join(coco_path, "annotations")
    ann_file = os.path.join(ann_dir, "instances_val2017.json")
    
    os.makedirs(coco_path, exist_ok=True)
    
    # 1. Validation Images (val2017.zip)
    if not os.path.exists(val_img_dir) or not os.listdir(val_img_dir):
        print(f"[Setup] 'val2017' not found in {coco_path}.")
        url = "http://images.cocodataset.org/zips/val2017.zip"
        if not download_and_extract(url, coco_path, "Validation Images"):
            raise RuntimeError("Failed to prepare validation images.")
    else:
        print(f"[Setup] Found 'val2017' images.")

    # 2. Annotations (annotations_trainval2017.zip)
    if not os.path.exists(ann_file):
        print(f"[Setup] 'instances_val2017.json' not found.")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        # annotations zip은 압축 풀면 annotations/ 폴더가 나옴. coco2017 바로 아래에 풀면 됨.
        if not download_and_extract(url, coco_path, "Annotations"):
            raise RuntimeError("Failed to prepare annotations.")
    else:
        print(f"[Setup] Found annotations.")

# ==========================================
# 2. 설정 및 모델 로드 함수
# ==========================================
def load_experiment_context(exp_dir):
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Config 찾기
    yaml_files = glob.glob(os.path.join(exp_dir, "*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml config file found in {exp_dir}")
    
    config_path = next((f for f in yaml_files if "config.yaml" in os.path.basename(f)), yaml_files[0])
    print(f"[Init] Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Weights 찾기
    pth_files = glob.glob(os.path.join(exp_dir, "*.pth"))
    weights_path = None
    
    for pth in pth_files:
        if "distilled_weights.pth" in os.path.basename(pth):
            weights_path = pth
            break
    
    if weights_path is None and pth_files:
        weights_path = pth_files[0] # Fallback

    if weights_path is None:
        raise FileNotFoundError(f"No .pth file found in {exp_dir}")

    print(f"[Init] Using weights: {weights_path}")
    cfg['STUDENT_WEIGHTS_PATH'] = weights_path
    
    return cfg

def load_student_model(cfg, device):
    backbone_name = cfg.get('STUDENT_MODEL_BACKBONE', 'convnext_tiny')
    text_dim = 640 # OpenCLIP ViT-B/32 output dim
    
    print(f"[Model] Building Student: {backbone_name}")
    
    if "multiscale" in backbone_name:
        base_name = backbone_name.replace("_multiscale", "")
        model = DistilledConvNeXtTinyMultiScale(text_dim=text_dim, backbone_name=base_name)
    else:
        model = DistilledConvNeXtTiny(text_dim=text_dim, backbone_name=backbone_name)

    weights_path = cfg['STUDENT_WEIGHTS_PATH']
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    return model.to(device)

def measure_efficiency(model, device, img_size):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    print("\n" + "="*40)
    print(" ⚡ Efficiency Benchmark")
    print("="*40)

    try:
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f" - Parameters : {params / 1e6:.2f} M")
        print(f" - FLOPs      : {macs * 2 / 1e9:.2f} G")
    except:
        print(" - FLOPs calc skipped")

    # FPS 측정
    for _ in range(10): _ = model(dummy_input)
    steps = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(dummy_input)
            if device == 'cuda': torch.cuda.synchronize()
    avg_time = (time.time() - start) / steps
    
    print(f" - Latency    : {avg_time * 1000:.2f} ms")
    print(f" - Throughput : {1/avg_time:.2f} FPS")
    print("="*40 + "\n")

@torch.no_grad()
def evaluate_retrieval(student, teacher, dataloader, device, tokenizer):
    student.eval()
    img_feats, txt_feats = [], []
    
    print("[Eval] Computing Recall@K...")
    
    # tqdm에 total 명시 및 예외처리 추가
    for batch in tqdm(dataloader):
        if batch is None: continue
        
        images, texts = batch
        images = images.to(device)
        
        # Student Image Emb
        s_img = student(images)
        s_img = s_img / s_img.norm(dim=1, keepdim=True)
        
        # Teacher Text Emb
        tokens = tokenizer(texts).to(device)
        t_txt = teacher.encode_text(tokens)
        t_txt = t_txt / t_txt.norm(dim=1, keepdim=True)
        
        img_feats.append(s_img.cpu())
        txt_feats.append(t_txt.cpu())

    if not img_feats:
        print("\n[Error] No valid data extracted. Please check dataset path or min_area_ratio.")
        return

    img_feats = torch.cat(img_feats, dim=0)
    txt_feats = torch.cat(txt_feats, dim=0)
    num_samples = img_feats.shape[0]
    
    print(f"[Eval] Total Valid Samples: {num_samples}")

    # Similarity Matrix: [N_txt, N_img]
    sim_matrix = txt_feats @ img_feats.T
    
    k_list = [1, 5, 10]
    metrics = {k: 0 for k in k_list}
    
    chunk_size = 100
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        sim_chunk = sim_matrix[i:end]
        targets = torch.arange(i, end).to(sim_chunk.device).view(-1, 1)
        
        _, topk_indices = sim_chunk.topk(max(k_list), dim=1)
        matches = (topk_indices == targets)
        
        for k in k_list:
            metrics[k] += matches[:, :k].any(dim=1).sum().item()

    print("-" * 30)
    for k in k_list:
        print(f" Recall@{k:<2} : {metrics[k]/num_samples*100:.2f}%")
    print("-" * 30)

def main():
    args = parse_args()
    
    # 1. Config 로드
    cfg = load_experiment_context(args.exp_dir)
    
    # Data Root Override (기본값 '.')
    if args.data_root:
        cfg['ROOT_DIR'] = args.data_root

    # =========================================================
    # 2. 데이터셋 자동 준비 (Val 없으면 다운로드)
    # =========================================================
    coco_dir_name = cfg.get('COCO_DIR_NAME', 'coco2017')
    prepare_coco_val(cfg['ROOT_DIR'], coco_dir_name)
    # =========================================================

    # 3. 데이터셋 로드
    coco_path = os.path.join(cfg['ROOT_DIR'], coco_dir_name)
    img_size = cfg.get('IMAGE_SIZE', 224)
    transform = get_clip_transform(img_size)
    
    # Validation용 Config (필터링 최소화)
    ds_cfg = {
        'MIN_AREA_RATIO': 0.0, # 검증 시에는 모든 객체 포함 권장
        'SMALL_ONLY': False,
        'TARGET_CLASS_NAMES': cfg.get('TARGET_CLASS_NAMES', []),
        'IMAGE_SIZE': img_size
    }

    print(f"[Data] Loading Validation Set from {coco_path}")
    val_dataset = COCORegionTextDataset(
        coco_dir=coco_path,
        cfg=ds_cfg,
        split='val2017', 
        ann='instances_val2017.json',
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['EVAL'].get('BATCH_SIZE', 32),
        shuffle=False,
        num_workers=cfg['EVAL'].get('NUM_WORKERS', 4),
        collate_fn=collate_fn
    )

    # 4. 모델 준비
    student = load_student_model(cfg, args.device)
    
    # Teacher (OpenCLIP) 로드
    teacher_name = cfg.get('TEACHER_MODEL', 'convnext_base_w')
    teacher_pretrain = cfg.get('TEACHER_PRETRAIN', 'laion2b_s13b_b82k')
    print(f"[Model] Loading Teacher: {teacher_name}")
    teacher, _, _ = open_clip.create_model_and_transforms(teacher_name, pretrained=teacher_pretrain)
    teacher.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(teacher_name)

    # 5. 실행
    measure_efficiency(student, args.device, img_size)
    evaluate_retrieval(student, teacher, val_loader, args.device, tokenizer)

if __name__ == "__main__":
    main()