import os
import zipfile
from pathlib import Path
import urllib.request
import sys

def download_and_setup_data(cfg):
    """
    COCO 및 KITTI 데이터셋을 다운로드하고 압축을 해제합니다.
    """
    ROOT = Path(cfg['ROOT_DIR'])
    ROOT.mkdir(parents=True, exist_ok=True) # ROOT 폴더 생성 보장
    
    # 1. 디렉토리 설정
    COCO_DIR = ROOT / cfg.get('COCO_DIR_NAME', 'coco')
    KITTI_DIR = ROOT / cfg.get('KITTI_DIR_NAME', 'kitti')
    
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    KITTI_DIR.mkdir(parents=True, exist_ok=True)

    # 2. URL 설정
    # COCO URL (Config에서 가져옴)
    coco_urls = cfg['DATA_URLS']
    
    # KITTI URL (ipynb 참고: AWS S3 미러 사용)
    kitti_img_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    kitti_lbl_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

    # 다운로드할 파일 목록 정의
    # Format: "파일명": {"url": URL, "dest": 압축해제경로, "check_path": 압축해제확인폴더}
    targets = {
        # --- COCO ---
        "train2017.zip": {
            "url": coco_urls['IMAGES_ZIP'],
            "dest": COCO_DIR,
            "check": COCO_DIR / "train2017"
        },
        "annotations_trainval2017.zip": {
            "url": coco_urls['ANN_ZIP'],
            "dest": COCO_DIR,
            "check": COCO_DIR / "annotations"
        },
        # --- KITTI ---
        "data_object_image_2.zip": {
            "url": kitti_img_url,
            "dest": KITTI_DIR,
            "check": KITTI_DIR / "training" / "image_2"
        },
        "data_object_label_2.zip": {
            "url": kitti_lbl_url,
            "dest": KITTI_DIR,
            "check": KITTI_DIR / "training" / "label_2"
        }
    }
    
    # 다운로드 진행률 표시 훅
    def reporthook(blocknum, blocksize, totalsize):
        readsoFar = blocknum * blocksize
        if totalsize > 0:
            percent = readsoFar * 100 / totalsize
            s = f"\rDownloading... {percent:5.1f}% ({readsoFar/1024/1024:.1f} MB)"
            sys.stderr.write(s)
            if readsoFar >= totalsize:
                sys.stderr.write("\n")
    
    # 3. 처리 루프 (다운로드 -> 압축 해제)
    for filename, info in targets.items():
        zip_path = ROOT / filename
        dest_dir = info['dest']
        check_path = info['check']

        # (1) 다운로드
        if not zip_path.exists():
            # 이미 압축이 풀려있다면 다운로드 스킵 (선택 사항)
            if check_path.exists():
                print(f"[Check] {filename} contents already exist. Skipping download.")
                continue

            print(f"\n[Download] Starting {filename}...")
            try:
                urllib.request.urlretrieve(info['url'], zip_path, reporthook=reporthook)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"\n[Error] Download failed for {filename}: {e}")
                if zip_path.exists():
                    os.remove(zip_path)
                continue # 다음 파일로 이동
        else:
            print(f"[Check] {filename} already exists. Skipping download.")

        # (2) 압축 해제
        if not check_path.exists():
            print(f"\n[Unzip] Extracting {filename} to {dest_dir}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(dest_dir)
                print("Extraction complete.")
            except zipfile.BadZipFile:
                print(f"[Error] {filename} is corrupted. Please delete it and run again.")
                # os.remove(zip_path) # 자동 삭제를 원하면 주석 해제
            except Exception as e:
                print(f"[Error] Unzip failed: {e}")
        else:
            print(f"[Check] {filename} is already extracted.")

    # 4. 최종 폴더 구조 확인
    print("\n--- Data Setup Report ---")
    
    coco_ready = (COCO_DIR / "train2017").exists() and (COCO_DIR / "annotations").exists()
    if coco_ready:
        print(f"✅ COCO Data Ready at: {COCO_DIR}")
    else:
        print(f"❌ COCO Data Incomplete")

    kitti_ready = (KITTI_DIR / "training" / "image_2").exists() and (KITTI_DIR / "training" / "label_2").exists()
    if kitti_ready:
        print(f"✅ KITTI Data Ready at: {KITTI_DIR}")
    else:
        print(f"❌ KITTI Data Incomplete")

    return COCO_DIR if coco_ready else None # 필요에 따라 리턴 값 조정