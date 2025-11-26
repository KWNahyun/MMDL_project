import os
import json
import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path

class Talk2CarDataset(data.Dataset):
    """
    Talk2Car Dataset Loader for MMDL Project.
    Assumes standard directory structure:
        root/
        ├── commands/
        │   ├── train_commands.json
        │   ├── test_commands.json
        │   └── val_commands.json
        └── images/
            ├── img_train_0.jpg
            └── ...
    """
    def __init__(self, data_dir, cfg, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        print(f"[Init] Talk2CarDataset ({split}) | Root: {self.data_dir}")

        # 1. 경로 설정
        self.img_dir = self.data_dir / "images"
        json_filename = f"{split}_commands.json"
        self.ann_path = self.data_dir / "commands" / json_filename

        # 2. 데이터 로드
        self.data = []
        if self.ann_path.exists():
            try:
                with open(self.ann_path, 'r') as f:
                    content = json.load(f)
                
                # {"commands": [...]} 구조 파싱
                if isinstance(content, dict) and "commands" in content:
                    self.data = self._filter_valid_data(content["commands"])
                    print(f"       Loaded {len(self.data)} samples from {json_filename}")
                else:
                    print(f"[Error] Invalid JSON format in {json_filename}. Expected 'commands' key.")
            except Exception as e:
                print(f"[Error] Failed to load annotations: {e}")
        else:
            print(f"[Error] Annotation file not found: {self.ann_path}")
            # raise FileNotFoundError(f"{self.ann_path} does not exist.") # 필요시 활성화

    def _filter_valid_data(self, raw_list):
        """필수 키(t2c_img, command, 2d_box)가 있는 데이터만 필터링"""
        valid_data = []
        for item in raw_list:
            if 't2c_img' in item and 'command' in item and '2d_box' in item:
                valid_data.append(item)
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
        except Exception:
            # 이미지 로드 실패 시 블랙 이미지 반환 (학습 중단 방지)
            img = Image.new('RGB', (1600, 900))
            
        w_orig, h_orig = img.size

        # Transform 적용
        if self.transform:
            img_tensor = self.transform(img)
        else:
            import torchvision.transforms as T
            img_tensor = T.ToTensor()(img)

        # 명령어 (Raw Text)
        command = item['command']

        # BBox 정규화 (Absolute [x,y,w,h] -> Normalized [0~1])
        bbox = item['2d_box']
        x, y, w, h = bbox
        
        norm_bbox = torch.tensor([
            x / w_orig,
            y / h_orig,
            w / w_orig,
            h / h_orig
        ], dtype=torch.float32)
        
        # 범위 클램핑 (0.0 ~ 1.0)
        norm_bbox = torch.clamp(norm_bbox, 0.0, 1.0)

        return img_tensor, command, norm_bbox

def talk2car_collate_fn(batch):
    """
    Batch: [ (img, cmd, bbox), ... ] -> (imgs_stack, cmd_list, bboxes_stack)
    """
    images, commands, bboxes = zip(*batch)
    return torch.stack(images, 0), list(commands), torch.stack(bboxes, 0)