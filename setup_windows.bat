@echo off
REM One-time setup: creates .venv and installs dependencies. No admin rights needed.
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment. Is Python 3.11 installed? Try: py -0p
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Full install failed. You can try the lite install: .venv\Scripts\python.exe -m pip install -r requirements-lite.txt
  exit /b 1
)
echo Setup complete. Start AyurBot with run_windows.bat
