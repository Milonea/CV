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
import yaml

def download_kaggle_dataset(kaggle_json_path, dataset_name, download_path):
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(kaggle_json_path)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_name,
        "-p", download_path,
        "--force"
    ]
    print(f"[INFO] Kaggle 데이터셋 다운로드 중: {dataset_name}")
    subprocess.run(cmd, check=True)
    zip_files = glob.glob(os.path.join(download_path, "*.zip"))
    if not zip_files:
        raise FileNotFoundError("[ERROR] 다운로드된 ZIP 파일을 찾을 수 없습니다.")
    zip_path = zip_files[0]
    print(f"[INFO] 압축 해제 중: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    print("[INFO] 다운로드 및 압축 해제 완료.")
    return zip_path
def load_config():
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
    project_root = os.path.dirname(os.path.abspath(__file__)) 

    setup_dir = os.path.join(project_root, "setup")
    os.makedirs(setup_dir, exist_ok=True)

    dataset_dir = os.path.join(project_root, "datasets", "css-data")
    os.makedirs(dataset_dir, exist_ok=True)

    kaggle_json_path = os.path.join(setup_dir, "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"[ERROR] Kaggle 인증 파일이 없습니다. (setup/kaggle.json)")

    dataset_name = load_config()
    
    zip_files = glob.glob(os.path.join(dataset_dir, "*.zip"))
    if zip_files:
        zip_path = zip_files[0]
        print(f"[INFO] 기존 ZIP 파일 존재, 다운로드 생략: {zip_path}")
    else:
        zip_path = download_kaggle_dataset(kaggle_json_path, dataset_name, dataset_dir)

    if not os.path.exists(os.path.join(dataset_dir, "train")):
        print("[INFO] 데이터셋 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("[INFO] 압축 해제 완료.")
    else:
        print("[INFO] 압축 해제된 데이터셋 폴더 존재, 최신 데이터 유지.")

    data_root = dataset_dir
    original_data_yaml_path = None
    found_train_dir = False
    
    if os.path.exists(os.path.join(data_root, "train")):
        found_train_dir = True
        print(f"[INFO] 기본 데이터 루트 발견: {data_root}")

    if not found_train_dir:
        search_pattern_1 = os.path.join(dataset_dir, "*", "train")
        search_pattern_2 = os.path.join(dataset_dir, "*", "*", "train")
        
        all_train_paths = glob.glob(search_pattern_1) + glob.glob(search_pattern_2)
        
        print(f"[DEBUG] 검색된 'train' 폴더 후보 경로: {all_train_paths}")

        if all_train_paths:

            new_data_root = os.path.dirname(all_train_paths[0]) 
            
            data_root = new_data_root 
            found_train_dir = True
            
            print(f"[INFO] 심층 데이터 루트 발견 및 설정: {data_root}")

    if not found_train_dir:
        raise FileNotFoundError(f"[CRITICAL ERROR] '{dataset_dir}' 내부에서 'train' 폴더를 찾을 수 없습니다. 데이터셋 구조를 확인하십시오.")

    original_data_yaml_path = glob.glob(os.path.join(data_root, "*.yaml"))
    
    if not original_data_yaml_path:
        original_data_yaml_path = glob.glob(os.path.join(os.path.dirname(data_root), "*.yaml"))
        
        if not original_data_yaml_path:
            print("[WARN] 데이터셋 루트에서 data.yaml 파일을 찾을 수 없습니다. 하드코딩된 클래스 정보를 사용합니다.")
            class_names = ['Person','vehicle']
            num_classes = len(class_names) 
        else:
            original_data_yaml_path = original_data_yaml_path[0]
    

    if isinstance(original_data_yaml_path, list) and original_data_yaml_path:
        original_data_yaml_path = original_data_yaml_path[0]
        
    if isinstance(original_data_yaml_path, str) and os.path.exists(original_data_yaml_path):
        try:
            with open(original_data_yaml_path, 'r', encoding='utf-8') as f:
                original_yaml = yaml.safe_load(f)
            class_names = original_yaml.get('names', [])
            num_classes = original_yaml.get('nc', len(class_names))
            print(f"[INFO] 기존 data.yaml에서 클래스 정보 ({num_classes}개)를 동적으로 추출했습니다.")
        except Exception as e:
            print(f"[ERROR] 기존 data.yaml 파일 파싱 오류: {e}. 하드코딩된 클래스 정보를 사용합니다.")
            class_names = ['Person','vehicle']
            num_classes = len(class_names)
    elif not 'class_names' in locals():
        class_names = ['Person','vehicle']
        num_classes = len(class_names)

    train_path = os.path.join(data_root, "train/images") 
    train_labels_path = os.path.join(data_root, "train/labels") 
    val_path = os.path.join(data_root, "valid/images") 
    val_labels_path = os.path.join(data_root, "valid/labels") 
    test_path = os.path.join(data_root, "test/images") 
    
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"[CRITICAL ERROR] 학습 라벨 폴더를 찾을 수 없습니다: {train_labels_path}. 데이터셋 구조를 확인해주세요. (train/images가 있다면, train/labels도 있어야 합니다.)")

    train_images = glob.glob(os.path.join(train_path, "*"))

    if not os.path.exists(val_path):
        if not train_images:
            print(f"[ERROR] '{train_path}'에서 학습용 이미지를 찾을 수 없습니다. 데이터셋 구조를 확인하세요. 학습을 건너뜁니다.")
            return
        else:
            print("[INFO] valid/images 폴더가 없어 train 데이터를 일부 복사합니다.")
            os.makedirs(val_path, exist_ok=True)
            os.makedirs(val_labels_path, exist_ok=True) 
            
            val_count = max(1, len(train_images)//10) 
            
            val_sample = random.sample(train_images, val_count)  # 10% 샘플
            
            for img_file in val_sample:
                shutil.copy(img_file, val_path)
                
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 현재 사용 장치: {device}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=480,
        batch=16,
        device=device,
        name=f"construction_yolov8_{device}",
        save_period=10,
        verbose=True,
    )

    print("[INFO] 학습 완료")
    print(f"[INFO] 결과 저장 위치: runs/train/construction_yolov8_{device}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()