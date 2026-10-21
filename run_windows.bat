@echo off
REM Starts AyurBot using the project virtual environment.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run setup_windows.bat first.
  exit /b 1
)
.venv\Scripts\python.exe app.py
