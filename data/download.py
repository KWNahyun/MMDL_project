import os
import zipfile
from pathlib import Path
import urllib.request
import sys

def download_and_setup_data(cfg):
    """
    COCO 데이터셋을 다운로드하고 압축을 해제합니다.
    """
    ROOT = Path(cfg['ROOT_DIR'])
    ROOT.mkdir(parents=True, exist_ok=True) # MMDL 폴더 생성 보장
    
    COCO_DIR = ROOT / cfg['COCO_DIR_NAME']
    COCO_DIR.mkdir(parents=True, exist_ok=True)

    # config의 DATA_URLS 가져오기
    urls = cfg['DATA_URLS']
    
    # URL과 저장할 파일명 매핑
    targets = {
        "train2017.zip": urls['IMAGES_ZIP'],
        "annotations_trainval2017.zip": urls['ANN_ZIP']
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
    
    # 1. 다운로드
    for filename, url in targets.items():
        zip_path = ROOT / filename
        
        if not zip_path.exists():
            print(f"\n[Download] Starting {filename}...")
            try:
                urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"\n[Error] Download failed for {filename}: {e}")
                # 실패한 파일이 부분적으로 남아있으면 삭제
                if zip_path.exists():
                    os.remove(zip_path)
                return None
        else:
            print(f"[Check] {filename} already exists. Skipping download.")

    # 2. 압축 해제
    
    # (1) 이미지 압축 해제
    images_zip = ROOT / "train2017.zip"
    if images_zip.exists() and not (COCO_DIR / "train2017").exists():
        print(f"\n[Unzip] Extracting {images_zip.name}...")
        try:
            with zipfile.ZipFile(images_zip, 'r') as zf:
                zf.extractall(COCO_DIR)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print("[Error] train2017.zip is corrupted. Please delete it and run again.")
            return None

    # (2) 주석 압축 해제
    ann_zip = ROOT / "annotations_trainval2017.zip"
    if ann_zip.exists() and not (COCO_DIR / "annotations").exists():
        print(f"\n[Unzip] Extracting {ann_zip.name}...")
        try:
            with zipfile.ZipFile(ann_zip, 'r') as zf:
                zf.extractall(COCO_DIR)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print("[Error] annotations_trainval2017.zip is corrupted. Please delete it and run again.")
            return None

    # 최종 확인
    if not (COCO_DIR / "train2017").exists() or not (COCO_DIR / "annotations").exists():
        print("\n[ERROR] COCO data setup incomplete (Missing extracted folders).")
        return None
        
    print("\n[Success] COCO data is ready.")
    return COCO_DIR