import cv2
from ultralytics import YOLO
import os
import glob
import numpy as np
import random 
import configparser

CONFIG_FILE = "setup/config_test.ini"
config = configparser.ConfigParser()

try:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {CONFIG_FILE}")
        
    config.read(CONFIG_FILE)
    
    MODEL_PATH = config['PATHS']['MODEL_PATH']
    IMAGE_DIR = config['PATHS']['IMAGE_DIR']
    
    CONF_THRESHOLD = config.getfloat('DETECTION', 'CONF_THRESHOLD')
    IOU_THRESHOLD = config.getfloat('DETECTION', 'IOU_THRESHOLD')
    
    DEFAULT_CLASSES = {}
    for key, value in config.items('CLASSES'):
        if key.startswith('class_'):
            try:
                cls_idx = int(key.split('_')[1])
                DEFAULT_CLASSES[cls_idx] = value
            except ValueError:
                print(f"경고: 잘못된 클래스 키 형식 '{key}'. 무시됩니다.")
                
except FileNotFoundError as e:
    print(f"오류: 설정 파일을 찾을 수 없습니다. {e}")
    exit()
except KeyError as e:
    print(f"오류: 설정 파일의 키 '{e}'가 누락되었습니다. INI 파일을 확인하세요.")
    exit()
except Exception as e:
    print(f"설정 파일 로드 중 예상치 못한 오류 발생: {e}")
    exit()

COLOR_PALETTE = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark Blue
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Red
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),# Silver
    (128, 128, 128) # Gray
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1 
TEXT_BORDER_THICKNESS = 2 
TEXT_COLOR = (255, 255, 255)
TEXT_BORDER_COLOR = (0, 0, 0)

def get_class_names(model, default_names):
    """모델(.pt)에서 클래스 이름을 가져오고, 없으면 기본 이름을 반환합니다."""
    try:
        if model.names and isinstance(model.names, dict) and len(model.names) > 0:
            print("  ▶ 모델 파일(.pt)에서 클래스 라벨을 성공적으로 로드했습니다.")
            return model.names
        else:
            print("  ▶ 모델 파일에 클래스 정보가 없거나 유효하지 않아 기본 라벨을 사용합니다.")
            return default_names
    except AttributeError:
        print("  ▶ 모델 객체에 names 속성이 없어 기본 라벨을 사용합니다.")
        return default_names
    except Exception as e:
        print(f"  ▶ 클래스 로드 중 오류 발생 ({e}). 기본 라벨을 사용합니다.")
        return default_names

def get_color_for_class(class_idx):
    """클래스 인덱스에 따라 색상을 반환합니다."""
    if class_idx < len(COLOR_PALETTE):
        return COLOR_PALETTE[class_idx]
    else:
        random.seed(class_idx) 
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

try:
    print(f"모델 로드 중: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        
    model = YOLO(MODEL_PATH)
    
    CLASS_NAMES = get_class_names(model, DEFAULT_CLASSES)
    
    search_path = os.path.join(IMAGE_DIR, '*')
    image_paths = glob.glob(search_path)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_paths:
        print(f"경고: '{IMAGE_DIR}' 폴더에서 이미지 파일이 발견되지 않았습니다. 프로그램을 종료합니다.")
        exit()
    
    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        print(f"\n--- 이미지 처리 시작: {file_name} ---")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"오류: 이미지 파일 읽기 실패: {image_path}. 스킵합니다.")
            continue
            
        results = model.predict(source=image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, save=False, verbose=False) 

        if results:
            result = results[0] 
            annotated_image = image.copy()
            boxes = result.boxes.cpu().numpy() 
            print(f"  탐지된 객체 수: {len(boxes)}")
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                cls_idx = int(box.cls[0])
                confidence = box.conf[0] 
                
                label = CLASS_NAMES.get(cls_idx, f"Unknown_{cls_idx}") 
                color = get_color_for_class(cls_idx)

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ({confidence:.2f})"
                (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
                
                cv2.rectangle(annotated_image, (x1, y1 - text_height - baseline - 10), (x1 + text_width, y1 - 10), color, -1)
                
                cv2.putText(annotated_image, text, (x1, y1 - baseline - 5),
                            FONT, FONT_SCALE, TEXT_BORDER_COLOR, TEXT_BORDER_THICKNESS)
                cv2.putText(annotated_image, text, (x1, y1 - baseline - 5),
                            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

            cv2.imshow(f"Detection Result: {file_name}", annotated_image)
            
            key = cv2.waitKey(0) 
            if key == 27: 
                break

    cv2.destroyAllWindows()
    
except FileNotFoundError as e:
    print(f"\n오류: {e}")
    print("경로 설정을 확인해 주세요.")
except Exception as e:
    print(f"\n예상치 못한 오류가 발생했습니다: {e}")