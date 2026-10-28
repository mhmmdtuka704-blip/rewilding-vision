@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP_AND_RUN.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\launch_edge.py
if errorlevel 1 pause
