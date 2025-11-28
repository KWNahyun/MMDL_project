# utils/dataset.py
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
from pathlib import Path
import torch
import os

# CLIP 표준 정규화 상수
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def get_clip_transform(image_size):
    """기본 CLIP 변환"""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def get_augmented_transform(image_size, cfg):
    """
    Albumentations 기반 강력한 증강 (Stage 1용)
    팀원 코드 스타일 반영
    """
    aug_cfg = cfg.get('AUGMENTATION', {})
    
    if not aug_cfg.get('USE_ALBUMENTATIONS', False):
        return get_clip_transform(image_size)
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg.get('HORIZONTAL_FLIP', 0.5)),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            ], p=aug_cfg.get('COLOR_JITTER', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=aug_cfg.get('SHIFT_SCALE', 0.05),
                scale_limit=aug_cfg.get('SHIFT_SCALE', 0.05),
                rotate_limit=aug_cfg.get('ROTATION', 5),
                p=0.3
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
            ], p=aug_cfg.get('NOISE_BLUR', 0.3)),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2()
        ])
    except ImportError:
        print("[Warning] Albumentations not installed. Using basic transform.")
        return get_clip_transform(image_size)

class COCORegionTextDataset(Dataset):
    def __init__(self, coco_dir, cfg, split="train2017", ann="instances_train2017.json", transform=None, max_images=None):
        
        self.coco_dir = Path(coco_dir)
        self.img_dir  = self.coco_dir / split
        self.ann_path = self.coco_dir / "annotations" / ann
        
        # 설정 로드
        self.transform = transform
        self.min_area_ratio = cfg['MIN_AREA_RATIO']
        self.small_only = cfg['SMALL_ONLY']
        self.target_class_names = set(cfg['TARGET_CLASS_NAMES'])
        self.image_size = cfg['IMAGE_SIZE']

        if not self.ann_path.exists():
            print(f"Annotation file not found at {self.ann_path}. Dataset cannot be initialized.")
            self.coco = None
            self.img_ids = []
            return

        self.coco = COCO(str(self.ann_path))

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.catid2name = {c['id']: c['name'] for c in cats}

        img_ids = self.coco.getImgIds()

        if self.target_class_names:
            target_ids = self.coco.getCatIds(catNms=list(self.target_class_names))
            filtered = []
            for img_id in img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=target_ids)
                if len(ann_ids) > 0:
                    filtered.append(img_id)
            img_ids = filtered

        if max_images:
            img_ids = img_ids[:max_images]

        self.img_ids = img_ids
        print(f"Using {len(self.img_ids)} images for training/evaluation.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if not self.coco: return {"image_path": "", "regions": []}

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]

        path = self.img_dir / img_info["file_name"]
        
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            return {"image_path": str(path), "regions": []}

        W, H = img.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        regions = []

        for ann in anns:
            cat = self.catid2name.get(ann['category_id'])
            if cat not in self.target_class_names:
                continue

            x,y,w,h = ann["bbox"]
            x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
            if x2<=x1 or y2<=y1: continue

            area = (x2-x1)*(y2-y1)
            area_ratio = area/(W*H)

            # 크기 필터링
            if self.small_only and area_ratio > 0.03: continue
            if not self.small_only and area_ratio < self.min_area_ratio: continue

            crop = img.crop((x1,y1,x2,y2))

            # 크기별 텍스트
            if area_ratio < 0.005: size = "a very small"
            elif area_ratio < 0.02: size = "a small"
            elif area_ratio < 0.08: size = "a medium"
            else: size = "a large"

            text = f"{size} {cat}"

            # 변환
            crop = crop.resize((self.image_size, self.image_size))
            t_img = self.transform(crop) if self.transform else T.ToTensor()(crop)

            regions.append({"image": t_img, "text": text})

        return {"image_path": str(path), "regions": regions}

def collate_fn(batch):
    """배치 병합"""
    imgs,texts=[],[]
    for b in batch:
        for r in b["regions"]:
            imgs.append(r["image"])
            texts.append(r["text"])
    if len(imgs)==0: return None
    return torch.stack(imgs,0), texts
