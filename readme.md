## 🚀 YOLOv8 기반 객체 탐지 학습 프로젝트

이 프로젝트는 **YOLOv8**을 사용하여 객체를 탐지하기 위한 자동화된 학습 파이프라인을 제공합니다. **Kaggle 데이터셋을 자동으로 다운로드**하고, 학습용/검증용 데이터를 구성하며, 최종적으로 YOLOv8 모델을 학습합니다.

-----

## 📦 주요 기능

  * **Kaggle 데이터셋 다운로드**
      * `setup/config.yaml`에 정의된 Kaggle 데이터셋 URL을 사용합니다.
      * ZIP 파일을 다운로드하고 자동으로 압축을 해제합니다.
  * **데이터셋 구성**
      * **`train`**, **`valid`**, **`test`** 폴더 구조를 구성합니다.
      * `valid/images`와 `valid/labels`는 **학습 데이터 일부를 샘플링**하여 자동 생성됩니다 (기본 **10%** 비율).
  * **클래스 정보 추출**
      * 데이터셋 내부의 \*\*`data.yaml`\*\*에서 클래스 정보를 자동으로 추출합니다.
      * 파일이 없을 경우 **하드코딩된 기본 클래스**를 사용할 수 있습니다.
  * **YOLOv8 학습**
      * **`yolov8s.pt`** 사전 학습 모델을 기반으로 학습을 진행합니다.
      * GPU 사용 가능 시 **CUDA를 자동 감지**하고 사용합니다.
      * 학습 로그 및 결과는 **`runs/train/`** 폴더에 저장됩니다.

-----

## ⚙️ 요구 사항

  * **Python $\geq 3.10$**
  * **Windows 환경에서 멀티프로세싱 안전 실행** 지원
  * **Kaggle 계정 및 API 키** (`setup/kaggle.json`에 저장)

-----

## 📚 필수 라이브러리 설치

> **⚠️ 주의:** CUDA 사용 시, PyTorch 설치 시점에 GPU 버전을 확인하고 적절히 설치해야 합니다.

```bash
# pip 최신 버전으로 업그레이드
python -m pip install --upgrade pip

# 주요 라이브러리 설치 (CUDA 12.1용 예시)
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib pyyaml requests
```

-----

## 🗂️ 프로젝트 구조

```
project_root/
│
├─ setup/
│   ├─ config.yaml         # Kaggle 데이터셋 URL 정의
│   └─ kaggle.json         # Kaggle API 인증 파일 (개인별 API Key 활용)
│
├─ datasets/
│   └─ css-data/           # Kaggle 데이터셋 다운로드 및 압축 해제 폴더 (실행 시 생성)
│
├─ train.py                 # 학습 실행 스크립트
└─ runs/
    └─ train/              # 학습 결과 저장 폴더 (실행 시 생성)
```

-----

## 🔧 실행 방법

### 1\. Kaggle 인증 파일 준비

`setup/` 폴더에 Kaggle API Key 파일을 저장합니다.

### 2\. Kaggle 데이터셋 URL 설정

`setup/config.yaml` 파일에서 다운로드할 데이터셋 URL을 설정합니다.

```yaml
# 기본 파일 구조
kaggle:
  dataset_url: "username/dataset-name"
```

### 3\. 학습 실행

`train.py` 스크립트를 실행하여 파이프라인을 시작합니다.

## ⚡ 주의 사항 및 설정

| 항목 | 설명 |
| :--- | :--- |
| **필수 폴더** | 학습을 위해서는 데이터셋 내에 **`train/labels`** 폴더가 반드시 존재해야 합니다. |
| **빈 데이터셋 처리** | 데이터셋 내 \*\*`train/images`\*\*가 비어있으면 학습을 건너뜁니다. |
| **검증 데이터 생성** | `valid/images`와 `valid/labels`는 **`train`** 데이터 일부를 **랜덤 샘플링**하여 생성됩니다. |
| **클래스 정의** | \*\*`data.yaml`\*\*이 없는 경우, **하드코딩된 기본 클래스**가 사용됩니다. |

### 📌 클래스 예시

하드코딩 기본 클래스는 다음과 같습니다:

```python
['Person','vehicle']
```

Tuning.py

## 📝 `setup/config.ini` 설정 가이드

```ini
[PATHS]
; train.py에서 설정한 학습 폴더 이름 (runs/train/아래 폴더 이름)
TRAIN_RUN_NAME = construction_yolov8_cuda
; 데이터 설정 파일 경로
VAL_DATA_PATH = setup/data.yaml

[TRAINING]
; 초기 모델 파일 이름
INITIAL_MODEL_NAME = best.pt

[HYPERPARAMETERS]
; 재학습 시 사용할 Epoch 수
NEW_EPOCHS = 30
; 재학습 시 사용할 초기 학습률 (lr0)
LEARNING_RATE = 0.005
; 재학습 시 데이터 증강 사용 여부 (True/False)
AUGMENT = True
```

## 🚀 사용법

### 1단계: 초기 학습 (`train.py` 실행)

`train.py`를 실행하여 데이터셋을 준비하고, `config.ini`에 설정된 이름(`TRAIN_RUN_NAME`)으로 초기 YOLOv8 모델을 학습시킵니다.

```bash
python train.py
```

> **결과:** `runs/train/construction_yolov8_cuda/weights/best.pt` 모델 파일과 `results.csv` 로그 파일이 생성됩니다.

-----

### 2단계: 성능 분석 및 튜닝 (`tuning.py` 실행)

`tuning.py`는 다음 3단계를 자동으로 수행합니다.

1.  **초기 모델 평가:** `best.pt` 파일을 로드하여 `mAP`, `mp`, `mr` 등의 초기 성능 지표를 출력합니다.
2.  **성능 시각화:** `results.csv` 파일을 읽어와 에폭별 Precision 및 Recall 추이를 Matplotlib 그래프로 시각화합니다.
3.  **튜닝 재학습 실행:** `config.ini`에 설정된 **`NEW_EPOCHS`** 및 **`LEARNING_RATE`** 등의 하이퍼파라미터를 사용하여 모델을 재학습(`model.train()`)하고, 최종 튜닝된 모델의 성능을 평가합니다.

<!-- end list -->

```bash
python tuning.py
```

> **결과:** 개선 모델이 생성되며, 개선된 성능 지표가 최종 출력됩니다.


Test.py

## 📦 실행 준비


## ⚙️ `config_Test.ini` 설정 파일

탐지 작업을 위한 모든 경로는 이 설정 파일을 통해 관리됩니다. 실행 전에 반드시 경로를 수정해야 합니다.

| 섹션      | 키               | 설명                                                                   | 예시 값                                                                 |
| :-------- | :--------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| `[PATHS]` | `MODEL_PATH`     | 학습된 YOLOv8 모델 파일(`.pt`)의 경로.                                | `runs/detect/construction_yolov8_cuda_tuned/weights/best.pt` |
| `[PATHS]` | `IMAGE_DIR`      | 탐지할 샘플 이미지가 포함된 폴더의 경로.                               | `sample`                                                                |
| `[DETECTION]`| `CONF_THRESHOLD` | 최소 신뢰도 임계값 (Confidence Score). 이 값 미만은 무시됩니다.        | `0.25`                                                                  |
| `[DETECTION]`| `IOU_THRESHOLD`  | Non-Maximum Suppression (NMS)의 IOU 임계값.                            | `0.7`                                                                   |
| `[CLASSES]`| `CLASS_0`        | 모델에 라벨이 없을 경우 사용할 기본 클래스 라벨 (인덱스:이름 형식). | `Class_A`                                                               |

## ▶️ 실행 방법

터미널에서 스크립트가 위치한 디렉터리로 이동하여 실행합니다.

```bash
python test.py
```

### 동작 순서

1.  스크립트는 `config_Test.ini`를 로드하여 설정 값을 가져옵니다.
2.  `MODEL_PATH`의 모델을 로드하고, 클래스 라벨을 설정합니다.
3.  `IMAGE_DIR` 내의 모든 이미지를 순회하며 탐지를 수행합니다.
4.  탐지된 객체를 시각화하여 **OpenCV 창**에 보여줍니다.
5.  이미지 창에서 아무 키나 누르면 다음 이미지로 넘어갑니다.
6.  **`ESC` 키**를 누르면 전체 실행을 종료하고 창을 닫습니다.

## 🎨 시각화 상세

  * **경계 상자**: 각 클래스 인덱스에 할당된 고유 색상으로 표시됩니다.
  * **라벨 텍스트**:
      * **배경**: 경계 상자와 동일한 색상의 사각형으로 채워집니다.
      * **글꼴**: 흰색으로 표시되어 모든 배경에서 잘 보입니다.
      * **테두리**: 가독성을 극대화하기 위해 검은색 테두리(`TEXT_BORDER_THICKNESS = 2`)가 적용됩니다.

## 📄 참고 자료

  * **YOLOv8 공식 문서:** [https://docs.ultralytics.com](https://docs.ultralytics.com)
  * **Kaggle API 문서:** [https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)