import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import random
import pandas as pd
import configparser 
from multiprocessing import freeze_support 

CONFIG_FILE = os.path.join('setup', 'config.ini')
config = configparser.ConfigParser()

try:
    config.read(CONFIG_FILE, encoding='utf-8') 
    
    TRAIN_RUN_NAME = config.get('PATHS', 'TRAIN_RUN_NAME')
    VAL_DATA_PATH = config.get('PATHS', 'VAL_DATA_PATH')
    
    INITIAL_MODEL_NAME = config.get('TRAINING', 'INITIAL_MODEL_NAME')
    TUNING_SUFFIX = config.get('TRAINING', 'TUNING_SUFFIX')
    
    NEW_EPOCHS = config.getint('HYPERPARAMETERS', 'NEW_EPOCHS')
    LEARNING_RATE = config.getfloat('HYPERPARAMETERS', 'LEARNING_RATE')
    AUGMENT = config.getboolean('HYPERPARAMETERS', 'AUGMENT')

except configparser.Error as e:
    print(f"FATAL ERROR: config.ini 파일을 읽는 중 오류 발생. 설정 파일 내용을 다시 확인하세요: {e}")
    exit()
except UnicodeDecodeError as e:
    print(f"FATAL ERROR: config.ini 파일 인코딩 오류 발생. 파일을 UTF-8로 저장했는지 확인하세요: {e}")
    exit()

INITIAL_MODEL_PATH = os.path.join("runs", "detect", TRAIN_RUN_NAME, "weights", INITIAL_MODEL_NAME) 
RESULTS_CSV_PATH = os.path.join("runs", "detect", TRAIN_RUN_NAME, "results.csv")
NEW_RUN_NAME = TRAIN_RUN_NAME + TUNING_SUFFIX
IMPROVED_MODEL_PATH = os.path.join("runs", "detect", NEW_RUN_NAME, "weights", INITIAL_MODEL_NAME) 

if __name__ == '__main__':
    freeze_support() 
    
    if not os.path.exists(VAL_DATA_PATH):
        print(f"⚠️ {VAL_DATA_PATH} 파일이 없습니다. train.py를 먼저 실행하여 데이터셋을 준비하세요.")
        dummy_data_yaml_content = f"""
train: dummy_path/train/images
val:   dummy_path/valid/images
nc: 2
names: ['object1', 'object2']
"""
        with open(VAL_DATA_PATH, 'w') as f:
            f.write(dummy_data_yaml_content.strip())

    print(f"\n--- 1. 초기 모델 ({INITIAL_MODEL_PATH}) 로드 및 평가 ---")
    
    initial_map, initial_mp, initial_mr = 0.60, 0.65, 0.55
    class DummyMetrics:
        def __init__(self, mAP, mp, mr):
            self.box = type('Box', (object,), {'map': mAP, 'mp': mp, 'mr': mr})
    metrics = DummyMetrics(initial_map, initial_mp, initial_mr)
    
    try:
        model = YOLO(INITIAL_MODEL_PATH) 
        metrics = model.val(data=VAL_DATA_PATH)
        print("\n✅ 초기 평가 결과:")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Mean Precision (mp): {metrics.box.mp:.4f}")
        print(f"Mean Recall (mr): {metrics.box.mr:.4f}")

    except FileNotFoundError as e:
        print(f"⚠️ 모델 또는 data.yaml 경로 오류: {e}")
        print(f"⚠️ train.py를 실행하여 모델 ({INITIAL_MODEL_PATH})과 data.yaml을 생성했는지 확인하세요.")
        print(f"⚠️ 임시 더미 지표: mAP50-95: {initial_map}, Mean Precision (mp): {initial_mp}, Mean Recall (mr): {initial_mr}")
        
    except AttributeError:
        print("⚠️ val() 결과에서 Mean Precision(mp) 또는 Mean Recall(mr) 속성을 찾을 수 없습니다. Ultralytics 버전을 확인하세요.")
        print(f"⚠️ 임시 더미 지표: mAP50-95: {initial_map}, Mean Precision (mp): {initial_mp}, Mean Recall (mr): {initial_mr}")
        
    print("\n--- 2. Matplotlib을 활용한 성능 시각화 ---")

    try:
        if os.path.exists(RESULTS_CSV_PATH):
            df = pd.read_csv(RESULTS_CSV_PATH, sep=',\s*', skipinitialspace=True, engine='python')
            
            epochs_list = df['epoch'].values
            sim_precision = df['metrics/precision(B)'].values
            sim_recall = df['metrics/recall(B)'].values
            print("  ▶ 실제 학습 로그(results.csv)를 로드하여 시각화합니다.")
        else:
            epochs_list = list(range(1, 21))
            sim_precision = [random.uniform(0.40, 0.50) + 0.015 * i for i in range(15)] + [0.71, 0.70, 0.69, 0.68, 0.67]
            sim_recall = [random.uniform(0.35, 0.45) + 0.01 * i for i in range(15)] + [0.60, 0.61, 0.60, 0.59, 0.58]
            sim_precision = sim_precision[:20]
            sim_recall = sim_recall[:20]
            print(f"  ▶ 로그 파일 없음 ({RESULTS_CSV_PATH}). 더미 데이터를 사용합니다.")

        plt.figure(figsize=(10, 5))
        
        plt.plot(epochs_list, sim_precision, label="Precision", marker='o', linestyle='-')
        plt.plot(epochs_list, sim_recall, label="Recall", marker='x', linestyle='--')
        
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Model Performance Trend (Precision vs. Recall)")
        plt.grid(True)
        plt.show() 

        print("📊 시각화 분석: 그래프를 보고 성능 개선 방안을 결정합니다.")

    except Exception as e:
        print(f"⚠️ 로그 파일 로드/시각화 중 오류 발생: {e}. pandas가 설치되었는지 확인하세요.")
        pass

    print("\n--- 3. 성능 향상 전략 적용: 데이터 증강(Augmentation) 후 재학습 ---")

    print(f"✅ 데이터 증강({AUGMENT})과 하이퍼파라미터 조정으로 재학습 시작...")
    
    try:
        model = YOLO(INITIAL_MODEL_PATH) 
        
        print(f"[INFO] Tuning 학습 시작: epochs={NEW_EPOCHS}, lr0={LEARNING_RATE}, augment={AUGMENT}")
        model.train(
            data=VAL_DATA_PATH, 
            epochs=NEW_EPOCHS, 
            imgsz=640, 
            augment=AUGMENT, 
            lr0=LEARNING_RATE,
            name=NEW_RUN_NAME
        )
        print("... 재학습 완료")
        
    except FileNotFoundError:
        print(f"⚠️ 초기 모델 ({INITIAL_MODEL_PATH})을 찾을 수 없어 재학습을 건너뜁니다.")
    except Exception as e:
        print(f"⚠️ 재학습 중 치명적인 오류 발생: {e}")
        pass


    print(f"파라미터: epochs={NEW_EPOCHS}, imgsz=640, augment={AUGMENT}, lr0={LEARNING_RATE}, name={NEW_RUN_NAME}")

    try:
        model_improved = YOLO(IMPROVED_MODEL_PATH) 
        improved_metrics = model_improved.val(data=VAL_DATA_PATH)
        
        print("\n✅ 개선된 모델 (Augmentation + Hyperparameter 적용) 최종 평가 결과:")
        print(f"mAP50-95 (개선): {improved_metrics.box.map:.4f}")
        print(f"Mean Precision (mp) (개선): {improved_metrics.box.mp:.4f}")
        print(f"Mean Recall (mr) (개선): {improved_metrics.box.mr:.4f}")
        print("🚀 성능 개선 여부를 확인하고, 목표 달성 시 배포합니다.")
    except FileNotFoundError:
        print(f"⚠️ 개선된 모델 ({IMPROVED_MODEL_PATH})을 찾을 수 없습니다. 재학습이 완료되었는지 확인하세요.")