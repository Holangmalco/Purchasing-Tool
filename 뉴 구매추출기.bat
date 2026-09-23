@echo off
chcp 65001 > nul

REM 1. 현재 bat 파일이 위치한 폴더로 이동 (어디서 실행하든 정상 동작)
cd /d "%~dp0"

REM 2. 가상환경이 해당 폴더에 있다면 활성화
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo ========================================================
echo 구매 문서 검증기를 실행합니다. 브라우저가 열릴 때까지 기다려주세요...
echo 사내 다른 PC에서 웹으로 접속하려면 아래 표시되는 Network URL을 사용하세요.
echo ========================================================

python -m streamlit run app.py
pause