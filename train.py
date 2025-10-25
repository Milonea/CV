import os
import zipfile
import json
import requests
import subprocess
from ultralytics import YOLO
import multiprocessing
import torch
import glob
import random 
import shutil 
import yaml # ⭐ YAML 파일 처리를 위해 추가

def download_kaggle_dataset(kaggle_json_path, dataset_name, download_path):
    """
    Kaggle API를 사용하여 데이터셋 다운로드 및 압축 해제
    """
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(kaggle_json_path)
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_name,
        "-p", download_path,
        "--force"
    ]
    print(f"[INFO] Kaggle 데이터셋 다운로드 중: {dataset_name}")
    subprocess.run(cmd, check=True)
    
    # 다운로드된 ZIP 파일 자동 탐지
    zip_files = glob.glob(os.path.join(download_path, "*.zip"))
    if not zip_files:
        raise FileNotFoundError("[ERROR] 다운로드된 ZIP 파일을 찾을 수 없습니다.")
    zip_path = zip_files[0]
    
    # 압축 해제
    print(f"[INFO] 압축 해제 중: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    print("[INFO] 다운로드 및 압축 해제 완료.")
    return zip_path
def load_config():
    """
    setup/config.yaml에서 Kaggle URL 읽기
    """
    config_path = os.path.join("setup", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config.yaml 파일이 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_name = config.get("kaggle", {}).get("dataset_url")
    if not dataset_name:
        raise ValueError("[ERROR] config.yaml에 kaggle.dataset_url이 지정되어 있지 않습니다.")
    return dataset_name
def main():
    # ===============================
    # 1️⃣ 경로 설정
    # ===============================
    # 프로젝트 루트 폴더 (현재 스크립트가 실행되는 위치)
    project_root = os.path.dirname(os.path.abspath(__file__)) 
    
    # 설정 파일 경로
    setup_dir = os.path.join(project_root, "setup")
    os.makedirs(setup_dir, exist_ok=True)

    # 데이터셋 저장 경로
    dataset_dir = os.path.join(project_root, "datasets", "css-data")
    os.makedirs(dataset_dir, exist_ok=True)

    # Kaggle 인증 파일 경로 확인
    kaggle_json_path = os.path.join(setup_dir, "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"[ERROR] Kaggle 인증 파일이 없습니다. (setup/kaggle.json)")

    # ===============================
    # 2️⃣ Kaggle 데이터셋 다운로드 / 최신 갱신
    # ===============================
    dataset_name = load_config()
    
    # ZIP 파일 존재 여부 확인
    zip_files = glob.glob(os.path.join(dataset_dir, "*.zip"))
    if zip_files:
        zip_path = zip_files[0]
        print(f"[INFO] 기존 ZIP 파일 존재, 다운로드 생략: {zip_path}")
    else:
        # 다운로드 및 압축 해제 함수 호출
        zip_path = download_kaggle_dataset(kaggle_json_path, dataset_name, dataset_dir)

    # ===============================
    # 3️⃣ 압축 해제 (최신 데이터 유지)
    # ===============================
    # 'train' 폴더가 없으면 압축 해제 진행
    if not os.path.exists(os.path.join(dataset_dir, "train")):
        print("[INFO] 데이터셋 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("[INFO] 압축 해제 완료.")
    else:
        print("[INFO] 압축 해제된 데이터셋 폴더 존재, 최신 데이터 유지.")

    # ===============================
    # 3.5️⃣ 실제 데이터셋 루트 폴더 동적 찾기 및 클래스 정보 추출 (수정된 부분)
    # ===============================
    data_root = dataset_dir
    original_data_yaml_path = None
    
    # 1. train 폴더를 포함하는 실제 데이터 루트 폴더 찾기
    if not os.path.exists(os.path.join(data_root, "train")):
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        found = False
        for subdir in subdirs:
            if os.path.exists(os.path.join(dataset_dir, subdir, "train")):
                data_root = os.path.join(dataset_dir, subdir)
                print(f"[INFO] 실제 데이터 루트 폴더 발견: {data_root}")
                found = True
                break
        
        if not found:
            raise FileNotFoundError(f"[CRITICAL ERROR] '{dataset_dir}' 내부에서 'train' 폴더를 찾을 수 없습니다. 데이터셋 구조를 확인하십시오.")
    
    # 2. 데이터셋 내부의 data.yaml에서 클래스 정보 추출
    original_data_yaml_path = glob.glob(os.path.join(data_root, "*.yaml"))
    
    if not original_data_yaml_path:
        # data.yaml 파일이 없으면 하드코딩된 정보를 사용하거나 오류 발생
        print("[WARN] 데이터셋 루트에서 data.yaml 파일을 찾을 수 없습니다. 하드코딩된 클래스 정보를 사용합니다.")
        class_names = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
        num_classes = 10
    else:
        original_data_yaml_path = original_data_yaml_path[0]
        try:
            with open(original_data_yaml_path, 'r', encoding='utf-8') as f:
                original_yaml = yaml.safe_load(f)
            class_names = original_yaml.get('names', [])
            num_classes = original_yaml.get('nc', len(class_names))
            print(f"[INFO] 기존 data.yaml에서 클래스 정보 ({num_classes}개)를 동적으로 추출했습니다.")
        except Exception as e:
            print(f"[ERROR] 기존 data.yaml 파일 파싱 오류: {e}. 하드코딩된 클래스 정보를 사용합니다.")
            class_names = ['Person','vehicle']
            num_classes = 10


    # ===============================
    # 4️⃣ val 폴더 자동 생성 (경로 설정 시 data_root 사용)
    # ===============================
    train_path = os.path.join(data_root, "train/images") 
    train_labels_path = os.path.join(data_root, "train/labels") 
    val_path = os.path.join(data_root, "valid/images") 
    val_labels_path = os.path.join(data_root, "valid/labels") 
    test_path = os.path.join(data_root, "test/images") 
    
    # 🚨 필수: 학습용 라벨 폴더가 존재하는지 확인
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"[CRITICAL ERROR] 학습 라벨 폴더를 찾을 수 없습니다: {train_labels_path}. 데이터셋 구조를 확인해주세요. (train/images가 있다면, train/labels도 있어야 합니다.)")

    # 학습용 이미지 목록을 먼저 가져옵니다.
    train_images = glob.glob(os.path.join(train_path, "*"))

    if not os.path.exists(val_path):
        if not train_images:
            # 학습용 이미지가 비어있을 경우 (이전 random.sample 오류 방지)
            print(f"[ERROR] '{train_path}'에서 학습용 이미지를 찾을 수 없습니다. 데이터셋 구조를 확인하세요. 학습을 건너뜁니다.")
            return # 이미지가 없으면 main 함수 종료
        else:
            print("[INFO] valid/images 폴더가 없어 train 데이터를 일부 복사합니다.")
            os.makedirs(val_path, exist_ok=True)
            os.makedirs(val_labels_path, exist_ok=True) 
            
            # 샘플링 수 계산: 전체의 10%, 최소 1개
            val_count = max(1, len(train_images)//10) 
            
            # random.sample 호출 (이제 train_images가 비어있지 않음이 보장됨)
            val_sample = random.sample(train_images, val_count)  # 10% 샘플
            
            for img_file in val_sample:
                # 1. 이미지 복사
                shutil.copy(img_file, val_path)
                
                # 2. 라벨 파일 찾기 및 복사
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                src_label_file = os.path.join(train_labels_path, base_name + ".txt") 
                dst_label_file = os.path.join(val_labels_path, base_name + ".txt")

                if os.path.exists(src_label_file):
                    shutil.copy(src_label_file, dst_label_file)
                else:
                    print(f"[WARN] 해당 이미지에 대한 라벨 파일 누락: {src_label_file}. 유효성 검사에서 이 이미지는 제외될 수 있습니다.")

            print(f"[INFO] 유효성 검사 데이터 {len(val_sample)}개 생성 완료 (이미지 및 라벨).")
    else:
        print("[INFO] 압축 해제된 데이터셋 폴더 존재, 최신 데이터 유지.")


    # ===============================
    # 5️⃣ data.yaml 생성 (동적 클래스 정보 반영)
    # ===============================
    # 클래스 리스트를 문자열로 변환
    names_str = str(class_names).replace("'", "")
    
    yaml_content = f"""
train: {train_path.replace(os.sep, '/')}
val:   {val_path.replace(os.sep, '/')}
test:  {test_path.replace(os.sep, '/')}
nc: {num_classes}
names: {names_str}
"""
    yaml_path = os.path.join(setup_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[INFO] data.yaml 생성 완료: {yaml_path}")

    # ===============================
    # 6️⃣ YOLOv8 모델 학습
    # ===============================
    # (이미지/라벨이 없는 경우 main 함수가 이미 종료되었으므로 안전합니다.)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 현재 사용 장치: {device}")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        name=f"construction_yolov8_{device}",
        save_period=10,
        verbose=True,
    )

    print("[INFO] 학습 완료")
    print(f"[INFO] 결과 저장 위치: runs/train/construction_yolov8_{device}")

# ===============================
# ✅ Windows 멀티프로세싱 안전 실행
# ===============================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
