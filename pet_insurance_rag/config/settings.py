# =========================================
# ⚙️ 설정 파일
# =========================================

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# === 기본 설정 ===
PROJECT_NAME = "펫보험 RAG 시스템"
VERSION = "1.0.0"
DESCRIPTION = "AI 기반 펫보험 상담 서비스"

# === API 설정 ===
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMB_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0.1
MAX_TOKENS = 1500

# === 데이터 처리 설정 ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_BACKEND = "faiss"
MEMORY_SIZE = 10

# === 파일 경로 설정 ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "embeddings_cache"

# 디렉토리 생성
for directory in [DATA_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# === CSV 파일 매핑 ===
CSV_FILES = {
    "삼성화재_애니펫": "삼성화재_반려견보험_애니펫.csv",
    "삼성화재_위풍댕댕": "삼성화재_위풍댕댕.csv",
    "삼성화재_착한펫": "삼성화재_착한펫보험.csv",
    "현대해상_굿앤굿우리펫보험": "현대해상_굿앤굿우리펫보험.csv", 
    "KB다이렉트_금쪽같은_펫보험": "KBdirect_금쪽같은_펫보험.csv",
    "메리츠화재_펫퍼민트": "meritz_펫퍼민트.csv",
    "DB손해보험_다이렉트_펫블리_반려견보험": "DB손해보험_다이렉트_펫블리_반려견보험.csv",
    "하나펫사랑보험": "하나펫사랑보험.csv"
}

# === Gradio 설정 ===
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False
GRADIO_DEBUG = False

# === 데이터베이스 설정 ===
FEEDBACK_DB_PATH = "user_feedback.db"
CHATBOT_DB_PATH = "ict_chatbot.db"

# === 로깅 설정 ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# === 환경변수 확인 및 설정 ===
def setup_environment():
    """환경변수 설정 및 확인"""
    
    # OpenAI API 키 확인
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # 백업: openaikey.txt 파일에서 읽기 시도
        api_key_file = BASE_DIR / "openaikey.txt"
        if api_key_file.exists():
            with open(api_key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            print("✅ OpenAI API 키를 백업 파일(openaikey.txt)에서 로드했습니다.")
        else:
            print("⚠️ OpenAI API 키가 설정되지 않았습니다.")
            print("   다음 중 하나를 선택하세요:")
            print("   1. .env 파일에 OPENAI_API_KEY=your-key 추가")
            print("   2. 환경변수 OPENAI_API_KEY 설정")
            print("   3. openaikey.txt 파일 생성")
            return False
    else:
        print("✅ OpenAI API 키를 환경변수에서 로드했습니다.")
    
    return True

# === 설정 검증 ===
def validate_config():
    """설정 검증"""
    errors = []
    
    # 필수 디렉토리 확인
    if not DATA_DIR.exists():
        errors.append(f"데이터 디렉토리가 없습니다: {DATA_DIR}")
    
    # CSV 파일 확인
    missing_files = []
    for company, filename in CSV_FILES.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            missing_files.append(f"{company}: {filename}")
    
    if missing_files:
        errors.append(f"누락된 CSV 파일들: {missing_files}")
    
    # API 키 확인
    if not setup_environment():
        errors.append("OpenAI API 키가 설정되지 않았습니다.")
    
    return errors

# === 설정 요약 출력 ===
def print_config():
    """현재 설정 출력"""
    print("=" * 50)
    print(f"🏷️  프로젝트: {PROJECT_NAME} v{VERSION}")
    print(f"📝 설명: {DESCRIPTION}")
    print("=" * 50)
    print(f"🤖 OpenAI 모델: {OPENAI_MODEL}")
    print(f"🔗 임베딩 모델: {OPENAI_EMB_MODEL}")
    print(f"🌡️  온도: {TEMPERATURE}")
    print(f"📏 청크 크기: {CHUNK_SIZE}")
    print(f"💾 벡터 백엔드: {VECTOR_BACKEND}")
    print(f"🌐 Gradio 포트: {GRADIO_SERVER_PORT}")
    print("=" * 50)
    print(f"📁 기본 디렉토리: {BASE_DIR}")
    print(f"📊 데이터 디렉토리: {DATA_DIR}")
    print(f"📝 로그 디렉토리: {LOGS_DIR}")
    print(f"🗄️  캐시 디렉토리: {CACHE_DIR}")
    print("=" * 50)
    
    # 설정 검증
    errors = validate_config()
    if errors:
        print("❌ 설정 오류:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ 모든 설정이 정상입니다!")
    print("=" * 50)

if __name__ == "__main__":
    print_config()