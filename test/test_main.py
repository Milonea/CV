import unittest
import json
import os
import shutil
import configparser
import sys
from unittest.mock import patch, MagicMock
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

CONFIG_PATH = os.path.join(project_root, 'config.ini')


class AppTest(unittest.TestCase):
    
    REAL_ORIGINAL_CONFIG_PATH = CONFIG_PATH + ".original_real"
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 전 초기 환경 구축 및 원본 config 백업"""
        cls.config_path = CONFIG_PATH
        cls.model_dir = os.path.join(project_root, 'model')
        
        cls.temp_original_config_path = cls.config_path + ".original" 

        if os.path.exists(cls.temp_original_config_path):
            os.remove(cls.temp_original_config_path)

        if os.path.exists(cls.config_path):
            shutil.copyfile(cls.config_path, cls.REAL_ORIGINAL_CONFIG_PATH)
        
        cls.initial_config = configparser.ConfigParser()
        cls.initial_config['YOLO_MODEL'] = {'model_path': 'yolov8s.pt', 'pretrained_model': 'yolov8s.pt'}
        cls.initial_config['SEARCH_SETTINGS'] = {'max_ingredients': '3', 'max_results': '5'}
        cls.initial_config['SERVER_SETTINGS'] = {'host': '0.0.0.0', 'port': '5000'}
        
        with open(cls.config_path, 'w', encoding='utf-8') as configfile:
            cls.initial_config.write(configfile)
            
        shutil.copyfile(cls.config_path, cls.temp_original_config_path)
        
        os.makedirs(cls.model_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 완료 후 생성된 파일 및 config 복원 정리"""
        
        if os.path.exists(cls.config_path):
            os.remove(cls.config_path)
        
        if os.path.exists(cls.temp_original_config_path):
            os.remove(cls.temp_original_config_path)
            
        if os.path.exists(cls.REAL_ORIGINAL_CONFIG_PATH):
            shutil.copyfile(cls.REAL_ORIGINAL_CONFIG_PATH, cls.config_path)
            os.remove(cls.REAL_ORIGINAL_CONFIG_PATH)
            
        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def setUp(self): 
        """각 테스트 시작 전 공통 설정"""
        
        try:
            import main as main_module
            self.app = main_module.app 
        except ImportError:
            self.app_client = MagicMock()
            
    def tearDown(self):
        """각 테스트 종료 후 정리"""
        pass

    @patch('main.YOLO') 
    @patch('main.Flask')
    def test_root_endpoint_success(self, MockFlask, MockYOLO):
        """루트 엔드포인트 (GET /)가 성공적인 응답을 반환하는지 테스트"""
        
        mock_app_instance = MagicMock()
        mock_app_instance.test_client.return_value = MagicMock()
        MockFlask.return_value = mock_app_instance
        client = mock_app_instance.test_client()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.data = json.dumps({"status": "ok", "message": "AI server is running"}).encode('utf-8')
        client.get.return_value = mock_response

        response = client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI server is running", response.data)

    @patch('main.YOLO')
    @patch('main.os.path.exists', return_value=True) 
    @patch('main.Flask')
    def test_detect_endpoint_success(self, MockFlask, MockExists, MockYOLO):
        """객체 감지 엔드포인트 (POST /detect)가 성공적으로 작동하는지 테스트"""
        
        mock_app_instance = MagicMock()
        mock_app_instance.test_client.return_value = MagicMock()
        MockFlask.return_value = mock_app_instance
        client = mock_app_instance.test_client()

        mock_result = MagicMock()
        mock_result.boxes.xyxy.tolist.return_value = [[100, 100, 200, 200]]
        mock_result.boxes.conf.tolist.return_value = [0.95]
        mock_result.names = {0: 'food', 1: 'drink'}
        mock_result.boxes.cls.tolist.return_value = [0]
        
        MockYOLO.return_value.predict.return_value = [mock_result]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = {
            "detections": [{
                "box": [100, 100, 200, 200],
                "label": "food",
                "confidence": 0.95
            }]
        }
        mock_response.data = json.dumps(mock_response.json).encode('utf-8')
        client.post.return_value = mock_response 
        
        data = {'file': (BytesIO(b'fake image data'), 'test_image.jpg')}

        response = client.post('/detect', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertIn('detections', response.json)


    @patch('main.Flask')
    def test_detect_endpoint_no_file(self, MockFlask):
        """파일 없이 POST /detect를 호출할 때 400 오류를 반환하는지 테스트"""
        
        mock_app_instance = MagicMock()
        mock_app_instance.test_client.return_value = MagicMock()
        MockFlask.return_value = mock_app_instance
        client = mock_app_instance.test_client()
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_error_json = {"error": "No image file provided"}
        mock_response.json = mock_error_json
        mock_response.data = json.dumps(mock_error_json).encode('utf-8')
        client.post.return_value = mock_response 

        response = client.post('/detect', data={})

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No image file provided", response.data)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)