# VIDA: Vision-Informed Deep Audio Feature Generation for Action Recognition

본 프로젝트는 오디오 정보가 결손된 비디오에서도 행동 인식 성능을 유지하기 위해, 비디오로부터 오디오 피처를 생성하고 이를 활용해 멀티모달 학습을 수행하는 두 가지 아키텍처(SVDMap, CAFMap)를 제안합니다.

---

## Repository 구조

```
├── SVDMap/                     # 아키텍처 1 관련 코드
│   └── model.ipynb             # 전체 파이프라인 실행 (학습 ~ 분류)
│   └── arc1_env.yml            # 아키텍처 1 Conda 환경 파일
│   └── data/                   # 실험에 필요한 입력 및 출력 파일 저장 디렉토리
│       ├── SAVLD.csv
│       └── pred_test.json
├── CAFMap/                     # 아키텍처 2 관련 코드 (지은아 여기 채워줘요!!)
│   └── ...
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
```

> 모든 실험은 위 환경에서 실행 가능합니다. `transformers`, `torch`, `torchaudio`, `opencv`, `scikit-learn`, `librosa` 등이 포함되어 있습니다.

---

## 데이터 다운로드 안내

본 프로젝트에서 사용하는 `.npy` 및 `.json` 데이터 파일은 총 용량 약 1.6GB 이상으로 GitHub에 직접 업로드하지 않았습니다.  
실행을 위해 필요한 파일 목록 및 위치는 아래 각 단계에 상세히 기술되어 있으며, 미포함된 데이터는 다음 두 가지 방식 중 하나로 접근하실 수 있습니다:

-  링크 (여기 데이터셋 링크 넣자요)
- `preprocessing/` 디렉토리 내 전처리 코드를 활용해 직접 생성

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

- 모든 실험은 `arc1_env.yml` 기반 Conda 환경에서 실행 가능
- `.npy` 파일은 용량 제한으로 GitHub에 포함되지 않았으며,  
  전처리 코드 또는 외부 링크를 통해 생성/수급 가능
- 실행 흐름은 모두 `SVDMap/model.ipynb` 내부 셀에서 단계별로 진행 가능

---

## 결과 요약 (여기 결과 숫자 수정해서 채워넣을까?)

| 조건                   | Accuracy |
|------------------------|----------|
| 오디오 없음            | 0.177    |
| 필터링된 오디오 사용   | **0.326**|
| GT 오디오 사용         | 0.343    |

---

## 🧑‍💻 팀원 소개

- 김지은 (2140010) – CAFMap 아키텍처 설계 및 구현  
- 윤서아 (2168019) – SVDMap 아키텍처 설계 및 구현  
- 장은성 (2271052) – 비디오 전처리 및 feature extractor 설계  

이화여자대학교, 2025년 캡스톤디자인

---

## 라이선스

본 프로젝트의 코드는 교육 및 비영리 연구 목적으로 공개됩니다.  
