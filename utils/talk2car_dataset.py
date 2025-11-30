# utils/talk2car_dataset.py
import os
import json
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
import numpy as np

def get_talk2car_augmentation(image_size, cfg):
    """
    Talk2Car 특화 증강 (자율주행 환경)
    - 날씨 변화, 조명 변화 시뮬레이션
    - BBox 보존을 위해 과도한 변형 제한
    """
    aug_cfg = cfg.get('AUGMENTATION', {})
    
    if not aug_cfg.get('USE_ALBUMENTATIONS', False):
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                       [0.26862954, 0.26130258, 0.27577711])
        ])
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg.get('HORIZONTAL_FLIP', 0.5)),
            
            # 날씨/조명 변화
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
            ], p=aug_cfg.get('COLOR_JITTER', 0.5)),
            
            # 날씨 효과
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
            ], p=aug_cfg.get('NOISE_BLUR', 0.3)),
            
            # 약간의 기하학적 변형 (BBox 깨지지 않도록 제한)
            A.ShiftScaleRotate(
                shift_limit=aug_cfg.get('SHIFT_SCALE', 0.05),
                scale_limit=aug_cfg.get('SHIFT_SCALE', 0.05),
                rotate_limit=aug_cfg.get('ROTATION', 5),
                border_mode=0,
                p=0.3
            ),
            
            A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=[], min_visibility=0.3))
    
    except ImportError:
        print("[Warning] Albumentations not installed. Using basic transform.")
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                       [0.26862954, 0.26130258, 0.27577711])
        ])

class Talk2CarDataset(data.Dataset):
    """
    Talk2Car Dataset Loader (Enhanced)
    - command_token 추가 (리더보드 제출용)
    - Albumentations 증강 지원
    """
    def __init__(self, data_dir, cfg, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.cfg = cfg
        
        # Transform 설정
        if transform is None:
            if split == 'train':
                self.transform = get_talk2car_augmentation(cfg['IMAGE_SIZE'], cfg)
            else:
                import torchvision.transforms as T
                self.transform = T.Compose([
                    T.Resize((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'])),
                    T.ToTensor(),
                    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                               [0.26862954, 0.26130258, 0.27577711])
                ])
        else:
            self.transform = transform
        
        print(f"[Init] Talk2CarDataset ({split}) | Root: {self.data_dir}")

        # 경로 설정
        self.img_dir = self.data_dir / "images"
        json_filename = f"{split}_commands.json"
        self.ann_path = self.data_dir / "commands" / json_filename

        # 데이터 로드
        self.data = []
        if self.ann_path.exists():
            try:
                with open(self.ann_path, 'r') as f:
                    content = json.load(f)
                
                if isinstance(content, dict) and "commands" in content:
                    self.data = self._filter_valid_data(content["commands"])
                    print(f"       Loaded {len(self.data)} samples from {json_filename}")
                else:
                    print(f"[Error] Invalid JSON format in {json_filename}.")
            except Exception as e:
                print(f"[Error] Failed to load annotations: {e}")
        else:
            print(f"[Error] Annotation file not found: {self.ann_path}")

    def _filter_valid_data(self, raw_list):
        """
        필수 키 확인 및 command_token 추가
        """
        valid_data = []
        missing_token_count = 0
        
        for item in raw_list:
            if 't2c_img' in item and 'command' in item and '2d_box' in item:
                # command_token 확인 (Test set 제출 필수)
                if 'command_token' not in item:
                    # command_token이 없으면 자동 생성 (img_id 기반)
                    item['command_token'] = f"token_{item.get('img_id', len(valid_data))}"
                    missing_token_count += 1
                
                valid_data.append(item)
        
        if missing_token_count > 0:
            print(f"[Warning] {missing_token_count} samples missing command_token (auto-generated)")
        
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 이미지 로드
        img_name = item['t2c_img']
        img_path = self.img_dir / img_name
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Failed to load {img_path}: {e}")
            img = Image.new('RGB', (1600, 900))
            
        w_orig, h_orig = img.size

        # BBox (Absolute)
        bbox = item['2d_box']
        x, y, w, h = bbox
        
        # -----------------------------------------------------------
        # [수정됨] Transform 호환성 처리 (Albumentations vs Torchvision)
        # -----------------------------------------------------------
        try:
            # 1. Albumentations 시도 (학습용)
            # - numpy 배열로 변환 필요
            # - image=, bboxes= 키워드 인자 필요
            # - 결과가 딕셔너리로 반환됨
            img_np = np.array(img)
            transformed = self.transform(image=img_np, bboxes=[[x, y, w, h]])
            img_tensor = transformed['image']
            
            # Augmentation으로 변형된 BBox 좌표 계산
            if len(transformed['bboxes']) > 0:
                tx, ty, tw, th = transformed['bboxes'][0]
                norm_bbox = torch.tensor([
                    tx / self.cfg['IMAGE_SIZE'],
                    ty / self.cfg['IMAGE_SIZE'],
                    tw / self.cfg['IMAGE_SIZE'],
                    th / self.cfg['IMAGE_SIZE']
                ], dtype=torch.float32)
            else:
                # Augmentation 중 BBox가 잘려나간 경우 (예: Crop) -> 원본 좌표 사용
                norm_bbox = torch.tensor([
                    x / w_orig, y / h_orig, w / w_orig, h / h_orig
                ], dtype=torch.float32)

        except (TypeError, KeyError):
            # 2. Torchvision 시도 (검증/테스트용)
            # - PIL Image를 직접 받음 (numpy 변환 불필요)
            # - 키워드 인자(image=) 사용 불가
            # - 텐서를 바로 반환
            img_tensor = self.transform(img)
            
            # 좌표는 원본 비율대로 정규화
            norm_bbox = torch.tensor([
                x / w_orig, y / h_orig, w / w_orig, h / h_orig
            ], dtype=torch.float32)
        # -----------------------------------------------------------

        # 범위 클램핑 (0.0 ~ 1.0)
        norm_bbox = torch.clamp(norm_bbox, 0.0, 1.0)

        # 명령어 및 토큰
        command = item['command']
        command_token = item['command_token']

        return img_tensor, command, norm_bbox, command_token

def talk2car_collate_fn(batch):
    """
    Batch 생성 (command_token 포함)
    """
    images, commands, bboxes, tokens = zip(*batch)
    return torch.stack(images, 0), list(commands), torch.stack(bboxes, 0), list(tokens)
