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

```bash
python train.py
```

  * **GPU 사용 시 자동 감지 및 사용**됩니다.
  * 학습 완료 후 결과는 다음 위치에 저장됩니다.
      * **GPU 환경:** `runs/train/construction_yolov8_cuda`
      * **CPU 환경:** `runs/train/construction_yolov8_cpu`

-----

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

-----

## 📄 참고 자료

  * **YOLOv8 공식 문서:** [https://docs.ultralytics.com](https://docs.ultralytics.com)
  * **Kaggle API 문서:** [https://github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)