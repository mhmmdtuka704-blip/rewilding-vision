@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\setup_model.py
if errorlevel 1 goto failed
call OPEN_IN_EDGE.bat
exit /b
:failed
echo Setup failed. Check the message above. Python 3.13 and internet are needed for first-time setup.
pause
exit /b 1
