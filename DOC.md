# 📘 프로젝트 문서: 2단계 지식 증류 기반 경량 Talk2Car 모델

## 1\. 프로젝트 개요 (Project Overview)

### 1.1. 배경 및 문제 정의

  * **자율주행 환경의 제약:** 자율주행 차량은 실시간성이 중요하므로 연산 자원과 메모리 사용에 제약이 있음. 거대 멀티모달 모델(예: CLIP-ViT-L/14)을 직접 탑재하기 어려움.
  * **Talk2Car 태스크의 난이도:** "저기 빨간 차 뒤에 멈춰"와 같은 자연어 명령을 이해하고, 복잡한 도로 상황에서 특정 객체(Referring Expression)를 정확히 지칭(Grounding)해야 함.
  * **데이터 부족:** Talk2Car 데이터셋만으로는 일반적인 시각-언어 관계를 학습하기에 데이터 양이 부족함.

### 1.2. 프로젝트 목표

  * **경량화:** 대형 OpenCLIP 모델(Teacher)의 지식을 경량 ConvNeXt-Tiny 모델(Student)로 이전(Distillation)하여 속도와 효율성을 확보.
  * **성능 유지:** 2단계 학습 전략을 통해 경량 모델임에도 불구하고 높은 수준의 Visual Grounding 성능(IoU, Accuracy) 달성.

-----

## 2\. 전체 파이프라인 (Pipeline Architecture)

프로젝트는 크게 두 단계(Stage)로 나뉘어 진행됩니다.

### **[Stage 1] Knowledge Distillation (지식 증류)**

  * **목표:** ConvNeXt-Tiny(Student)가 OpenCLIP-Base(Teacher)의 멀티모달 임베딩 공간을 모방하도록 학습.
  * **데이터:** COCO 2017 (방대한 일반 객체 이미지-텍스트 쌍).
  * **핵심 전략:**
      * **Region-based Learning:** 이미지를 통째로 넣지 않고, 객체 영역(Region)을 크롭하여 학습시킴으로써 "작은 객체"에 대한 인식 능력 강화.
      * **복합 Loss:** 단순 임베딩 일치뿐만 아니라 관계성(Relation)까지 학습.

### **[Stage 2] Talk2Car Fine-tuning (파인튜닝)**

  * **목표:** 자율주행 도메인에서 자연어 명령을 해석하여 Bounding Box 좌표를 예측.
  * **데이터:** Talk2Car (자율주행 이미지 + 복합 명령문).
  * **핵심 전략:**
      * **Train/Val Split:** 과적합 방지를 위한 데이터 분리.
      * **No Pooling & Transformer Head:** 위치 정보를 살리기 위해 Pooling을 제거하고, Transformer Decoder를 통해 텍스트가 이미지의 특정 위치를 조회(Attention)하도록 설계.
      * **L1 + GIoU Loss:** 정밀한 박스 좌표 보정.

-----

## 3\. 프로젝트 구조 (Project Structure)

```text
MMDL/
├── config/
│   └── default.yaml         # [설정] 모든 하이퍼파라미터, 경로, 모델 옵션 통합 관리
├── data/
│   └── download.py          # [데이터] COCO 데이터셋 자동 다운로드 및 압축 해제
├── models/
│   ├── student_model.py     # [모델] Student Encoder (Single/Multi-scale) 및 전체 통합 모델
│   └── grounding_head.py    # [모델] Projection / Transformer 기반 Grounding Head
├── utils/
│   ├── dataset.py           # [유틸] Stage 1용 COCO 데이터셋 (Region Crop)
│   ├── talk2car_dataset.py  # [유틸] Stage 2용 Talk2Car 데이터셋 (Img+Cmd+BBox)
│   ├── loss.py              # [유틸] 손실 함수 (DistillationLosses, Talk2CarLoss)
│   ├── training.py          # [유틸] 학습(Train), 평가(Eval), 추론(Inference) 함수 모음
│   ├── evaluation.py        # [유틸] 상세 분석용 (IoU 분포 등)
│   └── visualization.py     # [유틸] 결과 시각화 (BBox 그리기, Denormalization)
├── results/                 # [결과] 실험별 로그, 설정, 가중치, 시각화 결과 자동 저장
├── main.py                  # [실행] 전체 파이프라인 엔트리 포인트
└── README.md                # [문서] 프로젝트 설명
```

-----

## 4\. 상세 모듈 및 메서드 설명

### 4.1. Models (`models/`)

#### `student_model.py`

  * **`DistilledConvNeXtTiny`**:
      * `backbone`: `timm`의 `convnext_tiny`. `global_pool=''` 설정을 통해 $H \times W$ 피처맵을 유지(Stage 2용).
      * `forward(x)`: Stage 1용. Global Pooling 후 텍스트 차원으로 투영된 임베딩 반환.
  * **`DistilledConvNeXtTinyMultiScale`**:
      * ConvNeXt의 Stage 2, 3, 4 피처를 추출하고 융합(Fusion)하여 다양한 크기의 객체 정보를 포착.
  * **`Talk2CarModel`**:
      * Stage 2용 통합 모델. 학습된 Encoder와 `GroundingHead`를 결합.
      * `forward(images, text_emb)`: 이미지를 인코딩하고 텍스트 임베딩과 함께 Head에 입력하여 BBox 예측.

#### `grounding_head.py`

  * **`TransformerGroundingHead` (권장)**:
      * Transformer Decoder 구조 사용.
      * **Query:** 텍스트 임베딩 / **Key, Value:** 이미지 피처맵.
      * 텍스트 명령이 이미지의 어느 공간(Spatial)에 집중해야 하는지 학습.
  * **`ProjectionGroundingHead`**:
      * 단순 MLP 구조. 이미지와 텍스트 벡터를 Concat하여 예측 (Baseline용).

### 4.2. Utils (`utils/`)

#### `dataset.py` (Stage 1)

  * **`COCORegionTextDataset`**:
      * COCO 주석을 파싱하여 객체별 Bounding Box를 잘라냄(Crop).
      * 작은 객체(Small)와 큰 객체(Large)를 구분하고, 이에 맞는 텍스트 캡션("a small car") 생성.
  * **`get_augmented_transform`**: Albumentations 라이브러리를 활용한 강력한 데이터 증강.

#### `talk2car_dataset.py` (Stage 2)

  * **`Talk2CarDataset`**:
      * `train_commands.json` 등을 파싱하여 `(이미지, 명령어, BBox)` 쌍을 제공.
      * BBox 좌표를 이미지 크기(1600x900)로 나누어 **0\~1 범위로 정규화(Normalization)**.
      * 경로 자동 탐색 기능 포함.

#### `loss.py`

  * **`DistillationLosses` (Stage 1)**:
      * `L_clip` (InfoNCE): 텍스트-이미지 대조 학습.
      * `L_sim` (Similarity-KD): Teacher와 Student의 Logit 분포(관계) 일치 (KL Divergence).
      * `L_cos` (Post-Cosine): 임베딩 벡터의 방향 일치.
  * **`Talk2CarLoss` (Stage 2)**:
      * `L1 Loss`: 예측 좌표와 정답 좌표의 절대 거리 차이 최소화.
      * `GIoU Loss`: 박스 겹침 정도(IoU)를 직접 최적화하고, 겹치지 않은 경우에도 거리를 좁힘.

#### `training.py`

  * **`train_epoch`**: Stage 1 학습 루프. Teacher는 Frozen 상태로 임베딩만 생성.
  * **`fine_tune_epoch`**: Stage 2 학습 루프. L1+GIoU Loss 역전파.
  * **`evaluate_talk2car`**: Validation 셋에 대해 **Average IoU**와 **AP50 (IoU \>= 0.5)** 성능 측정.
  * **`inference_and_visualize`**: Test 셋 추론 및 결과 이미지 저장.

-----

## 5\. 실행 가이드 (How to Run)

### 기본 실행 (전체 파이프라인)

```bash
python main.py --stage all
```

  * Stage 1 학습 → 가중치 저장 → Stage 2 학습 → 결과 저장 순으로 자동 진행.

### 단계별 실행

  * **Stage 1만 실행:** `python main.py --stage 1`
  * **Stage 2만 실행:** `python main.py --stage 2` (자동으로 최신 `distilled_weights.pth`를 찾아 로드함)

### 시각화 및 예측 파일 생성

```bash
python main.py --stage test --visualize --generate_predictions
```

  * `test` 모드로 실행하여 `predictions.json`을 생성하고, 결과 이미지를 `results/.../visualizations/`에 저장.

-----

## 6\. 성능 지표 (Metrics)

  * **Stage 1:**
      * **Recall@K:** 텍스트 쿼리에 대해 올바른 이미지를 상위 K개 안에 찾았는지 비율. (Small/Large 객체별로 분석)
  * **Stage 2:**
      * **Average IoU:** 예측 박스와 정답 박스의 교집합 비율 평균.
      * **AP50 (Accuracy@0.5):** IoU가 0.5 이상인 예측의 비율 (리더보드 기준).

이 문서를 바탕으로 프로젝트 보고서나 깃허브 README를 보강하시면 됩니다.