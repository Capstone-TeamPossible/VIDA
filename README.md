# VIDA: Vision-Informed Deep Audio Feature Generation for Action Recognition

본 연구는 오디오 모달리티가 결손된 비디오 기반 행동 인식 환경에서, 시각 정보를 활용하여 의미적으로 정합한 오디오 피처를 생성하는 새로운 멀티모달 학습 프레임워크인 VIDA (Vision-Informed Deep Audio Feature Generation for Action Recognition)를 제안한다. 제안하는 프레임워크는 SVDMap과 CAFMap의 두 아키텍처로 구성되며, 모두 Transformer 기반 시각 피처 추출기와 LSTM 기반 오디오 생성기를 중심으로 의미 정합성을 학습 과정에 통합하여 멀티모달 표현의 신뢰성과 활용도를 높인다. Moments-in-Time 데이터셋을 기반으로 한 실험 결과, 실제 오디오가 없는 상황에서도 생성된 오디오 피처를 활용했을 때 SVDMap은 약 35%, CAFMap은 약 31%의 정확도를 기록하였다. 이는 기존 오디오 미포함 모델 대비 안정적인 성능 향상을 보인 것으로, MiT 데이터셋의 SOTA 정확도인 53%에는 도달하지 못했으나, 오디오 결손 환경에서도 유의미한 행동 인식 성능을 확보할 수 있음을 시사한다.

---

## Repository 구조

```
├── SVDMap/                     # 아키텍처 1 코드
│   └── model.ipynb             # 전체 파이프라인 실행 (학습 ~ 분류)
│   └── arc1_env.yml            # 아키텍처 1 Conda 환경 파일
│   └── data/                   # 실험에 필요한 입력 및 출력 파일 저장 디렉토리
│       ├── SAVLD.csv
│       └── pred_test.json
├── CAFMap/                     # 아키텍처 2 코드
│   └── model.ipynb             # 전체 파이프라인 실행 (학습 ~ 분류)
│   └── arc2_env.yaml           # 아키텍처 2 Conda 환경 파일
├── data-example/               # 오디오 및 비디오 데이터 전처리 코드
├── data-preprocessing/         # 오디오 및 비디오 데이터 전처리 코드
├── feature-extraction/         # 오디오 및 비디오 피처 전처리 코드
└── README.md                   # 설명 파일
```

---

## 환경 설정 (How to build)

```bash
# arc1 Conda 환경 생성
conda env create -f arc1_env.yml
conda activate arc1_env

# arc2 Conda 환경 생성
conda env create -f arc2_env.yaml
conda activate arc1_env
```

> 모든 실험은 위 환경에서 실행 가능합니다. `transformers`, `torch`, `torchaudio`, `opencv`, `scikit-learn`, `librosa` 등이 포함되어 있습니다.

---

## 데이터 다운로드 안내

본 프로젝트에서 사용되는 각 모듈의 산출물 `.npy` 및 `.json` 데이터 파일은 총 용량 약 1.6GB 이상으로 GitHub에 직접 업로드하지 않았습니다.  
실행을 위해 필요한 파일 목록 및 위치는 아래 각 단계에 상세히 기술되어 있습니다. 

비디오 행동 인식을 위한 원본 데이터의 크기는 275GB로 실험 환경에서 직접 다운로드가 필요합니다.
원본 데이터는 다음 링크에서 접근 가능하며 실험 재현을 위해서는 `preprocessing/` 디렉토리 내 코드를 활용한 전처리가 필요합니다.

[Moments in Time Dataset](http://moments.csail.mit.edu/)

---

## ▶️ Architecture 1: SVDMap

본 아키텍처는 LSTM으로 오디오 피처를 생성한 뒤, AST 모델을 통해 multi-label prediction을 수행하고, Semantic Dictionary와의 의미 정합성 기반 필터링을 통해 유효한 피처만을 행동 인식에 활용합니다.

---

### 1. LSTM 기반 오디오 피처 생성

- 필요한 파일:  
  `audio_filtered_train.npy`, `audio_filtered_val.npy`, `audio_filtered_test.npy`  
  `rgb_filtered_train.npy`, `rgb_filtered_val.npy`, `rgb_filtered_test.npy`  
  `flow_filtered_train.npy`, `flow_filtered_val.npy`, `flow_filtered_test.npy`  
  → 총 9개

- 실행 파일:  
  `SVDMap/model.ipynb`

- 출력:  
  `gen_audio_train.npy`, `gen_audio_val.npy`, `gen_audio_test.npy`

---

### 2. 오디오 기반 행동 분류

- 필요한 파일:  
  위에서 생성된 `gen_audio_*.npy` 3개

- 실행 파일:  
  `SVDMap/model.ipynb`

- 출력:  
  Test accuracy, Confusion Matrix

---

### 3. AST 기반 멀티라벨 예측

- 필요한 파일:  
  `audio_waveform_test.npy`

- 실행 파일:  
  `SVDMap/model.ipynb`

- 출력:  
  `pred_test.json`, `ast_pred_test_top5.json`

---

### 4. IoU 기반 의미 정합성 필터링

- 필요한 파일:  
  `SAVLD.csv`, `pred_test.json`

- 실행 파일:  
  `SVDMap/model.ipynb`

- 출력:  
  `audio_filtered_train_selected.npy`, `...val_selected.npy`, `...test_selected.npy` 등 총 9개

---

### 5~6. Filtering 기반 Action Classifier 학습

- 필요한 파일:  
  Filtering을 거친 selected `.npy` 총 9개

- 실행 파일:  
  `SVDMap/model.ipynb`

- 출력:  
  Final classification accuracy using filtered audio

---

## ▶️ Architecture 2: CAFMap

> ✍️ [작성 중]  
Transformer 기반 캡션 생성 및 semantic attention mapping 구조에 관한 내용은 추후 업데이트될 예정입니다.

---

## 재현 시 주의사항

- 모든 실험은 `arc1_env.yml` 또는 `arc2_env.yaml` 기반 Conda 환경에서 실행 가능
- 각 네트워크의 중간 산출물 `.npy` 파일은 용량 제한으로 GitHub에 포함되지 않았으며,  
  전처리 코드 또는 외부 링크를 통해 생성 가능
- 실행 흐름은 모두 `SVDMap/model.ipynb` 또는 `CAFMap/model.ipynb` 내부 셀에서 단계별로 진행 가능

---

## 결과

- **SVDMap**
  
| Model                   | Accuracy |
|------------------------|----------|
| Baseline          | 17.7%    |
| Filtered Audio   | **35.6%**|
| Ground Truth Audio         | 34.3%    |

- **CAFMap**
  
| Model                   | Accuracy |
|------------------------|----------|
| Baseline           | 22.5%    |
| Generated Audio   | **30.7%%**|
| Ground Truth Audio         | 33.9%    |

---

## 👩🏻‍💻 팀원 소개

- **김지은** (2140010) – CAFMap 아키텍처 설계 및 구현  
- **윤서아** (2168019) – SVDMap 아키텍처 설계 및 구현  
- **장은성** (2271052) – 비디오 전처리 및 feature extractor 설계  

이화여자대학교, 2025년 캡스톤디자인

---

## 라이선스

본 프로젝트의 코드는 교육 및 비영리 연구 목적으로 공개됩니다.  
