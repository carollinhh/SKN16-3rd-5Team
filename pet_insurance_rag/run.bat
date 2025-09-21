# =========================================
# 🚀 빠른 실행 스크립트 
# =========================================

@echo off
echo ========================================
echo   🐾 펫보험 RAG 시스템 실행
echo ========================================

REM Python 버전 확인
python --version
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았거나 PATH에 없습니다.
    pause
    exit /b 1
)

REM 가상환경 활성화 (선택사항)
if exist "venv\Scripts\activate.bat" (
    echo 🔧 가상환경 활성화 중...
    call venv\Scripts\activate.bat
)

REM 필요한 패키지 설치
echo 📦 필수 패키지 설치 중...
pip install -r config\requirements.txt
if %errorlevel% neq 0 (
    echo ❌ 패키지 설치 실패
    pause
    exit /b 1
)

REM 시스템 실행
echo 🚀 시스템 시작...
python main.py

pause