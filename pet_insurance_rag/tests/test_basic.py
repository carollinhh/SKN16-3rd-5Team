# =========================================
# 🧪 기본 시스템 테스트
# =========================================

import unittest
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class TestSystemSetup(unittest.TestCase):
    """시스템 설정 테스트"""
    
    def test_config_import(self):
        """설정 모듈 import 테스트"""
        try:
            from config.settings import PROJECT_NAME, VERSION
            self.assertEqual(PROJECT_NAME, "펫보험 RAG 시스템")
            self.assertEqual(VERSION, "1.0.0")
        except ImportError:
            self.fail("설정 모듈을 import할 수 없습니다.")
    
    def test_project_structure(self):
        """프로젝트 구조 테스트"""
        required_dirs = [
            "src",
            "config", 
            "data",
            "notebooks",
            "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"{dir_name} 디렉토리가 없습니다.")
    
    def test_required_files(self):
        """필수 파일 존재 테스트"""
        required_files = [
            "main.py",
            "README.md",
            "src/__init__.py",
            "config/__init__.py",
            "config/settings.py",
            "config/requirements.txt"
        ]
        
        for file_name in required_files:
            file_path = project_root / file_name
            self.assertTrue(file_path.exists(), f"{file_name} 파일이 없습니다.")

class TestDataProcessing(unittest.TestCase):
    """데이터 처리 모듈 테스트"""
    
    def test_module_import(self):
        """모듈 import 테스트"""
        try:
            from src.data_processing import PetInsuranceGuard, LangChainEmbeddingsManager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"데이터 처리 모듈 import 실패: {e}")
    
    def test_pet_insurance_guard(self):
        """펫보험 가드 클래스 테스트"""
        try:
            from src.data_processing import PetInsuranceGuard
            guard = PetInsuranceGuard()
            
            # 텍스트 정제 테스트
            clean_text = guard.clean_text("  반려동물 보험\n\n약관  ")
            self.assertEqual(clean_text, "반려동물 보험 약관")
            
            # 관련성 필터링 테스트
            relevant = guard.filter_relevant_content("강아지 치료비 보장")
            self.assertTrue(relevant)
            
            irrelevant = guard.filter_relevant_content("일반적인 텍스트")
            self.assertFalse(irrelevant)
            
        except Exception as e:
            self.fail(f"PetInsuranceGuard 테스트 실패: {e}")

class TestRAGFunctions(unittest.TestCase):
    """RAG 함수 모듈 테스트"""
    
    def test_module_import(self):
        """모듈 import 테스트"""
        try:
            from src.rag_functions import DualAgentSystem, UserFeedbackEvaluator
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"RAG 함수 모듈 import 실패: {e}")
    
    def test_feedback_evaluator(self):
        """피드백 평가자 테스트"""
        try:
            from src.rag_functions import UserFeedbackEvaluator
            
            # 메모리 DB로 테스트
            evaluator = UserFeedbackEvaluator(db_path=":memory:")
            
            # 평가 테스트
            scores = {
                "정확성": 4,
                "완성도": 3,
                "명확성": 5,
                "실용성": 4,
                "친근함": 3
            }
            
            result = evaluator.evaluate_response(
                question="테스트 질문",
                answer="테스트 답변",
                scores=scores,
                comments="테스트 코멘트"
            )
            
            self.assertTrue(result["success"])
            self.assertIn("feedback_id", result)
            
        except Exception as e:
            self.fail(f"UserFeedbackEvaluator 테스트 실패: {e}")

class TestGradioInterface(unittest.TestCase):
    """Gradio 인터페이스 모듈 테스트"""
    
    def test_module_import(self):
        """모듈 import 테스트"""
        try:
            from src.gradio_interface import GradioInterfaceManager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Gradio 인터페이스 모듈 import 실패: {e}")

def run_basic_tests():
    """기본 테스트 실행"""
    print("🧪 기본 시스템 테스트 실행...")
    print("=" * 50)
    
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    suite.addTests(loader.loadTestsFromTestCase(TestSystemSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestGradioInterface))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ 모든 테스트 통과!")
    else:
        print(f"❌ {len(result.failures)} 실패, {len(result.errors)} 오류")
        
        if result.failures:
            print("\n실패한 테스트:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
        
        if result.errors:
            print("\n오류가 발생한 테스트:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('\n')[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)