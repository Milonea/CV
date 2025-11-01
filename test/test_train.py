import unittest
import os
import shutil
import configparser
from unittest.mock import MagicMock
from datetime import datetime
import sys
import unittest.mock

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
retrain_module_path = os.path.join(project_root, 'ai_retrain')
sys.path.insert(0, retrain_module_path)

try:
    import train as target_retrain
except ImportError as e:
    print(f"FATAL ERROR: train.py 파일을 임포트할 수 없습니다. 오류: {e}")
    sys.exit(1)
    
sys.path.pop(0)

class RetrainTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 전 초기 환경 구축 및 원본 config 백업"""
        cls.config_path = os.path.join(project_root, 'config.ini')
        cls.original_config_path = cls.config_path + ".original" 
        cls.model_dir = os.path.join(project_root, 'model')
        
        if os.path.exists(cls.config_path):
            os.remove(cls.config_path)
        if os.path.exists(cls.original_config_path):
            os.remove(cls.original_config_path)
            
        cls.initial_config = configparser.ConfigParser()
        cls.initial_config['YOLO_MODEL'] = {'model_path': 'yolov8s.pt', 'pretrained_model': 'yolov8s.pt'}
        cls.initial_config['SEARCH_SETTINGS'] = {'max_ingredients': '3', 'max_results': '5'}
        cls.initial_config['TRAINING_SETTINGS'] = {'epochs': '50', 'img_size': '640'}
        
        with open(cls.config_path, 'w', encoding='utf-8') as configfile:
            cls.initial_config.write(configfile)
            
        shutil.copyfile(cls.config_path, cls.original_config_path)
        
        os.makedirs(cls.model_dir, exist_ok=True)
        cls.mock_run_dir = os.path.join(retrain_module_path, 'runs/detect/custom_food_retrain/weights')
        os.makedirs(cls.mock_run_dir, exist_ok=True)
        with open(os.path.join(cls.mock_run_dir, 'best.pt'), 'w') as f:
            f.write("mock_model_data")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 완료 후 생성된 파일 및 config 복원 정리"""
        
        backup_config_path = cls.config_path + ".bak"
        
        if os.path.exists(cls.original_config_path):
            try:
                shutil.copyfile(cls.original_config_path, cls.config_path)
            except Exception:
                pass 
        
        if os.path.exists(cls.original_config_path):
            os.remove(cls.original_config_path)
        if os.path.exists(backup_config_path):
            os.remove(backup_config_path)
        
        shutil.rmtree(cls.model_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(retrain_module_path, 'runs'), ignore_errors=True)

    def test_run_retraining_updates_config(self):
        
        with unittest.mock.patch.object(target_retrain, 'YOLO') as MockYOLO, \
             unittest.mock.patch.object(target_retrain, 'datetime') as MockDatetimeClass:
            
            fixed_time_str = '20251110_103000'
            mock_datetime_instance = unittest.mock.MagicMock()
            mock_datetime_instance.strftime.return_value = fixed_time_str
            MockDatetimeClass.now.return_value = mock_datetime_instance
            
            mock_results = MagicMock()
            mock_results.save_dir = os.path.join(retrain_module_path, 'runs/detect/custom_food_retrain')
            MockYOLO.return_value.train.return_value = mock_results
            
            original_cwd = os.getcwd()
            os.chdir(retrain_module_path)
            target_retrain.run_retraining() 
            os.chdir(original_cwd) 

            updated_config = configparser.ConfigParser()
            with open(self.config_path, 'r', encoding='utf-8') as f:
                 updated_config.read_file(f)
            
            expected_path = os.path.join('model', f'best_{fixed_time_str}.pt')
            
            self.assertEqual(updated_config.get('YOLO_MODEL', 'model_path'), expected_path,
                             "config.ini의 model_path가 버전 정보로 업데이트되지 않았습니다.")
                             
            versioned_model_path = os.path.join(self.model_dir, f'best_{fixed_time_str}.pt')
            self.assertTrue(os.path.exists(versioned_model_path), 
                            "버전별 모델 파일이 model 폴더에 생성되지 않았습니다.")

    def test_run_retraining_failure_restores_config(self):

        with unittest.mock.patch.object(target_retrain, 'YOLO') as MockYOLO, \
             unittest.mock.patch.object(target_retrain, 'datetime') as MockDatetimeClass:
            
            fixed_time_str = '20251110_103000'
            mock_datetime_instance = unittest.mock.MagicMock()
            mock_datetime_instance.strftime.return_value = fixed_time_str
            MockDatetimeClass.now.return_value = mock_datetime_instance
            
            mock_results = MagicMock()
            mock_results.save_dir = os.path.join(retrain_module_path, 'runs/detect/custom_food_retrain')
            MockYOLO.return_value.train.return_value = mock_results
            
            original_cwd = os.getcwd()
            os.chdir(retrain_module_path)
            
            with unittest.mock.patch('builtins.open') as MockOpen:
                
                MockOpen.side_effect = IOError("Simulated write failure")
                
                with self.assertRaises(Exception):
                    target_retrain.run_retraining()
                    
            os.chdir(original_cwd) 
            
            config_path = os.path.normpath(os.path.join(retrain_module_path, '../config.ini'))
            
            updated_config = configparser.ConfigParser()
            with open(config_path, 'r', encoding='utf-8') as f:
                updated_config.read_file(f)

            self.assertEqual(updated_config.get('YOLO_MODEL', 'model_path'), 'yolov8s.pt',
                             "설정 파일 저장 실패 후, 백업 파일로 복원이 제대로 이루어지지 않았습니다.")
            
            backup_config_path = os.path.normpath(os.path.join(retrain_module_path, '../config.ini.bak'))
            self.assertFalse(os.path.exists(backup_config_path),
                             "설정 파일 복원 후 백업 파일(.bak)이 정리되지 않고 남아 있습니다.")
                
if __name__ == '__main__':
    unittest.main()