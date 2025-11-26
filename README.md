# 💡 2단계 지식 증류를 통한 경량 Talk2Car 멀티모달 모델 개발

## 프로젝트 소개
본 프로젝트는 자율주행 환경에서 필수적인 **속도**와 **가벼움**을 확보하기 위해, 대형 OpenCLIP 교사 모델의 지식을 **경량 ConvNeXt-Tiny** 학생 모델로 이전하는 **2단계 지식 증류(Knowledge Distillation)** 방법론을 제안하고 구현합니다[cite: 2, 3, 12, 19].

### 핵심 목표
1.  **1단계 (Distillation):** ConvNeXt-Tiny 모델을 대규모 COCO 데이터셋을 활용하여 OpenCLIP-Base 모델의 CLIP 멀티모달 공간에 안정적으로 정렬합니다.
2.  **2단계 (Fine-tuning):** 1단계에서 정렬된 경량 모델을 복잡한 **Talk2Car 데이터셋**으로 파인튜닝하여 자율주행 명령 이해(Region Grounding) 태스크에 특화된 모델을 개발합니다.

## 🛠️ 프로젝트 구조

프로젝트는 모듈화되어 있으며, 모든 설정은 `config/default.yaml` 파일을 통해 관리됩니다.
```
MMDL/
├── config/
│   └── default.yaml         # 프로젝트 설정 (경로, 모델 H/W, 하이퍼파라미터, Head Type)
├── data/
│   └── download.py          # COCO 데이터셋 다운로드 및 압축 해제
├── models/
│   ├── student_model.py     # DistilledConvNeXtTiny 및 Talk2CarModel (통합 모델) 정의
│   └── grounding_head.py    # [New] Projection/Transformer Grounding Head 정의
├── utils/
│   ├── dataset.py           # Stage 1용 COCORegionTextDataset 정의
│   ├── talk2car_dataset.py  # [New] Stage 2용 Talk2CarDataset 및 파싱 로직 정의
│   ├── loss.py              # 복합 Loss (InfoNCE, SimKD, Post-Cosine) 정의
│   └── training.py          # Stage 1(증류) 및 Stage 2(파인튜닝) 훈련/평가 루프
├── results/                 # [New] 실험 결과 저장소 (main.py 실행 시 자동 생성)
│   └── result_YYYYMMDD_...  # 실행 시점별 로그(log), 설정(yaml), 가중치(pth) 저장
├── main.py                  # 메인 실행 스크립트 (단계별 실행, 로깅, 실험 관리)
├── coco2017/                # stage 1용 COCO 데이터셋
├── Talk2Car/                # stage 2용 Talk2Car 데이터셋
│       ├── predictions.json            
│       ├── commands/            
│       └── images/         
└── README.md
```
## ⚙️ 시작하기 (Getting Started)

### 1. 환경 설정

필요한 Python 라이브러리를 설치합니다.

``` bash
pip install -r requirements.txt
```
### 2. 설정 파일 확인
config/default.yaml 파일을 열어 데이터 경로 (ROOT_DIR), 모델 경로, 배치 크기 등을 환경에 맞게 조정합니다.

### 3. 데이터 준비 (1단계 COCO)
`main.py`를 실행하면 `data/download.py`가 자동으로 호출되어 COCO 2017 데이터셋 (이미지 및 주석)을 다운로드하고 `MMDL/coco2017` 경로에 준비합니다.

- 주의: COCO 데이터셋은 18GB 이상의 용량을 차지합니다.

### 4. 데이터 준비 (2단계 Talk2Car)
https://github.com/talk2car/Talk2Car 의 README.md 파일을 참고하여 Talk2Car 데이터셋을 다운로드하고 설정합니다.

### 5. 훈련 및 평가 실행
`main.py`를 실행하여 훈련 및 평가를 진행합니다. `--stage` 인자로 1 또는 2를 지정하여 각 단계별로 실행할 수 있습니다.
``` bash
python3 main.py --stage 1
python3 main.py --stage 2
python3 main.py --stage all
```