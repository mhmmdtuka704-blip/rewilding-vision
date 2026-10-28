@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP_AND_RUN.bat first to create this project's environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address=127.0.0.1 --server.headless=false
if errorlevel 1 pause
