# =========================================
# 🐾 펫보험 RAG 시스템 메인 실행 스크립트
# =========================================

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 설정 로드
try:
    from config.settings import setup_environment, print_config, validate_config
    from src.data_processing import initialize_data_processing
    from src.rag_functions import initialize_rag_functions
    from src.gradio_interface import initialize_gradio_interface
    from src.evaluation import create_evaluation_system
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("필요한 패키지를 설치해주세요:")
    print("pip install -r config/requirements.txt")
    sys.exit(1)

def main():
    """메인 실행 함수"""
    
    print("🚀 펫보험 RAG 시스템 시작!")
    print("=" * 60)
    
    # 1. 설정 확인 및 출력
    print("📋 1단계: 설정 확인")
    print_config()
    
    # 설정 검증
    errors = validate_config()
    if errors:
        print("❌ 설정 오류로 인해 실행을 중단합니다.")
        return False
    
    # 2. 환경 설정
    print("\n 2단계: 환경 설정")
    if not setup_environment():
        print("❌ 환경 설정 실패")
        return False
    print("✅ 환경 설정 완료")
    
    # 3. 데이터 처리 모듈 초기화
    print("\n📊 3단계: 데이터 처리 모듈 초기화")
    try:
        data_components = initialize_data_processing()
        print("✅ 데이터 처리 모듈 초기화 완료")
    except Exception as e:
        print(f"❌ 데이터 처리 모듈 초기화 실패: {e}")
        return False
    
    # 4. 평가 시스템 초기화
    print("\n📊 4단계: 평가 시스템 초기화")
    try:
        user_feedback_evaluator, ab_test_evaluator, real_time_monitor = create_evaluation_system()
        print("✅ 평가 시스템 초기화 완료")
        print(f"  - 사용자 피드백 평가기 초기화")
        print(f"  - A/B 테스트 평가기 초기화")
        print(f"  - 실시간 모니터 초기화")
    except Exception as e:
        print(f"❌ 평가 시스템 초기화 실패: {e}")
        return False

    # 5. RAG 함수 모듈 초기화 (평가 시스템과 함께)
    print("\n🤖 5단계: RAG 함수 모듈 초기화")
    try:
        rag_components = initialize_rag_functions(data_components['filtered_retrievers'])
        # 평가 시스템을 RAG 컴포넌트에 추가
        rag_components['user_feedback_evaluator'] = user_feedback_evaluator
        rag_components['ab_test_evaluator'] = ab_test_evaluator
        rag_components['real_time_monitor'] = real_time_monitor
        print("✅ RAG 함수 모듈 초기화 완료")
    except Exception as e:
        print(f"❌ RAG 함수 모듈 초기화 실패: {e}")
        return False

    # 6. Gradio 인터페이스 초기화
    print("\n🌐 6단계: Gradio 인터페이스 초기화")
    try:
        interface_components = initialize_gradio_interface(
            dual_agent=rag_components['dual_agent'],
            feedback_evaluator=user_feedback_evaluator,
            company_vector_stores=data_components['company_vector_stores']
        )
        print("✅ Gradio 인터페이스 초기화 완료")
    except Exception as e:
        print(f"❌ Gradio 인터페이스 초기화 실패: {e}")
        return False

    # 7. 시스템 정보 출력
    print("\n📈 7단계: 시스템 정보")
    print(f"📊 처리된 회사 수: {len(data_components['company_vector_stores'])}")
    print(f"🔍 검색기 수: {len(data_components['filtered_retrievers'])}")
    
    total_docs = sum(
        len(vs.docstore._dict) 
        for vs in data_components['company_vector_stores'].values()
        if hasattr(vs, 'docstore')
    )
    print(f"📄 총 문서 수: {total_docs}")
    
    # 8. 인터페이스 실행
    print("\n🌐 8단계: 웹 인터페이스 실행")
    print("=" * 60)
    print("🎉 모든 초기화가 완료되었습니다!")
    print("📱 웹 브라우저에서 인터페이스에 접속하세요.")
    print("🔄 종료하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    try:
        # Gradio 인터페이스 실행
        interface_components['interface_manager'].launch(
            share=False,
            debug=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 시스템을 종료합니다...")
        return True
    except Exception as e:
        print(f"\n❌ 인터페이스 실행 오류: {e}")
        return False

def quick_test():
    """간단한 테스트 실행"""
    print("🧪 빠른 테스트 모드")
    
    # 설정만 확인
    from config.settings import validate_config
    errors = validate_config()
    
    if errors:
        print("❌ 설정 오류:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ 설정 검증 통과!")
        return True

if __name__ == "__main__":
    # 커맨드라인 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = quick_test()
    else:
        success = main()
    
    sys.exit(0 if success else 1)