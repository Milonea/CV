import os
import requests
import configparser
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from urllib.parse import quote
load_dotenv()
config = configparser.ConfigParser()
try:
    with open('config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)
    if not config.sections(): 
        print("FATAL ERROR: config.ini 파일을 읽었으나 내용이 비어있습니다. 파일 내용을 확인해 주세요.")
        exit()
    # ---------------------------------------------
        
except Exception as e:
    print(f"FATAL ERROR: config.ini 파일 로드 실패. 오류: {e}")
    exit()
API_KEY = os.environ.get("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, 
            static_url_path='/static',
            template_folder=TEMPLATE_FOLDER) 
app.config['TEMPLATES_AUTO_RELOAD'] = True 
            
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = config.get('YOLO_MODEL', 'model_path') 
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"FATAL ERROR: YOLO 모델 로드 실패 - {MODEL_PATH}. 오류: {e}")
    exit()

def get_max_ing():
    """config.ini에서 max_ingredients 값을 안전하게 로드합니다."""
    try:
        return config.getint('SEARCH_SETTINGS', 'max_ingredients')
    except (configparser.NoOptionError, ValueError):
        return 3 

def recognize_ingredients(image_path):
    """YOLOv8 모델을 사용하여 이미지에서 식재료를 인식하고 리스트를 반환합니다."""
    results = yolo_model(image_path, imgsz=640, conf=0.5) 
    recognized_ingredients = set()
    
    for r in results:
        names = r.names
        for box in r.boxes:
            class_id = int(box.cls[0])
            ingredient_name = names[class_id]
            recognized_ingredients.add(ingredient_name)
                 
    return sorted(list(recognized_ingredients))

def search_recipes(ingredients, max_ing):
    """Google Custom Search API를 사용하여 레시피를 검색합니다."""
    
    if not API_KEY or not SEARCH_ENGINE_ID:
        return [{'title': 'API 설정 오류', 'link': '키를 .env 파일에 설정해주세요.'}]
    try:
        max_res = config.getint('SEARCH_SETTINGS', 'max_results')
    except (configparser.NoOptionError, ValueError):
        max_res = 5
    
    query_ingredients = ingredients[:max_ing] 
    query = " ".join(query_ingredients) + " 레시피"

    encoded_query = quote(query) 

    URL = (f"https://www.googleapis.com/customsearch/v1?"
           f"key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={encoded_query}&num={max_res}")
    
    try:
        response = requests.get(URL)
        response.raise_for_status() 
        search_results = response.json()
        
        items = []
        if 'items' in search_results:
            for item in search_results['items']:
                items.append({
                    'title': item.get('title'),
                    'link': item.get('link')
                })
        return items
    except requests.exceptions.RequestException as e:
        print(f"웹 검색 오류: {e}")
        return [{'title': '웹 검색 실패', 'link': f'오류 발생: {e}'}]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            ingredients = recognize_ingredients(filepath)
            
            return render_template('index.html', 
                                   ingredients=ingredients, 
                                   uploaded_file=filename)

    return render_template('index.html', ingredients=None)

@app.route('/search', methods=['POST'])
def search_ingredients():
    """사용자가 체크박스로 선택한 식재료를 받아 레시피를 검색합니다."""
    
    selected_ingredients = request.form.getlist('ingredients') 
    
    max_ing = get_max_ing()
    
    recipes = []
    if selected_ingredients:
        recipes = search_recipes(selected_ingredients, max_ing)
        
    return render_template('search_results.html', 
                           selected_ingredients=selected_ingredients,
                           recipes=recipes,
                           max_ingredients=max_ing) 

if __name__ == '__main__':
    app.run(debug=True)