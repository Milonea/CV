import os
import shutil
import configparser
from ultralytics import YOLO
from datetime import datetime
import sys

CONFIG_FILE = "../config.ini" 
config = configparser.ConfigParser()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH_ABS = os.path.normpath(os.path.join(CURRENT_DIR, CONFIG_FILE))
BACKUP_FILE = CONFIG_PATH_ABS + ".bak"

try:
    with open(CONFIG_PATH_ABS, 'r', encoding='utf-8') as f:
        config.read_file(f)

    if not config.has_section('YOLO_MODEL'):
        raise configparser.NoSectionError('YOLO_MODEL')
        
except FileNotFoundError:
    print(f"FATAL ERROR: 설정 파일 '{CONFIG_PATH_ABS}'을 찾을 수 없습니다.")
    sys.exit(1)
except configparser.NoSectionError as e:
    print(f"FATAL ERROR: 설정 파일에 '{e.section}' 섹션이 없습니다. 파일을 확인해 주세요.")
    sys.exit(1)

DATA_YAML_PATH = "data.yaml" 
PRETRAINED_MODEL = config.get('YOLO_MODEL', 'pretrained_model', fallback='yolov8s.pt') 
EPOCHS = config.getint('TRAINING_SETTINGS', 'epochs', fallback=50) 
IMG_SIZE = config.getint('TRAINING_SETTINGS', 'img_size', fallback=640)

DEPLOY_MODEL_DIR = os.path.join("..", "model")

def run_retraining():
    print("--- 1. YOLOv8 모델 재학습 시작 ---")
    
    model = YOLO(PRETRAINED_MODEL) 

    results = model.train(
        data=DATA_YAML_PATH, 
        epochs=EPOCHS, 
        imgsz=IMG_SIZE,
        name='custom_food_retrain' 
    )

    print("--- 2. 학습 완료 및 버전 파일 생성 ---")
    
    best_weight_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    if not os.path.exists(best_weight_path):
        print("FATAL ERROR: 최적 모델 파일(best.pt)을 찾을 수 없습니다.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"best_{timestamp}.pt"
    
    DEPLOY_MODEL_PATH_VERSIONED = os.path.join(DEPLOY_MODEL_DIR, versioned_filename)
    
    print(f"--- 모델 배포: {best_weight_path} -> {DEPLOY_MODEL_PATH_VERSIONED} ---")
    
    os.makedirs(DEPLOY_MODEL_DIR, exist_ok=True)
    
    shutil.copyfile(best_weight_path, DEPLOY_MODEL_PATH_VERSIONED)
    
    print("--- config.ini 파일 업데이트 ---")
    
    try:
        shutil.copyfile(CONFIG_PATH_ABS, BACKUP_FILE)
    except Exception as e:
        print(f"FATAL ERROR: 설정 파일 백업에 실패했습니다. 업데이트를 중단합니다.: {e}")
        return

    new_model_path_for_config = os.path.join("model", versioned_filename)
    
    config.set('YOLO_MODEL', 'model_path', new_model_path_for_config)
    
    try:
        with open(CONFIG_PATH_ABS, 'w', encoding='utf-8') as configfile:
            config.write(configfile)
            
        os.remove(BACKUP_FILE)
        print("✅ 설정 파일 백업 파일 삭제 완료.")

    except Exception as e:
        print(f"ERROR: 설정 파일 저장 중 오류 발생. 백업 파일로 복원 시도 중...: {e}")
        try:
            shutil.copyfile(BACKUP_FILE, CONFIG_PATH_ABS)
            print("✅ 설정 파일 복원 완료.")
        except Exception as restore_e:
            print(f"FATAL ERROR: 설정 파일 복원에도 실패했습니다. 수동 확인이 필요합니다.: {restore_e}")
            
        raise

    print(f"✅ 모델 재학습 및 배포 완료. 'config.ini'가 {new_model_path_for_config}로 업데이트되었습니다.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__))) 
    run_retraining()