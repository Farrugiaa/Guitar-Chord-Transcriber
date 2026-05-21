@echo off
setlocal

echo ============================================================
echo  Guitar Chord Extractor — Setup
echo ============================================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Please install Python 3.12 from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%
echo.

:: Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists, skipping creation.
)
echo.

:: Install PyTorch (CPU build — works on any machine, no GPU required)
echo Installing PyTorch (CPU build)...
venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
echo.

:: Install remaining dependencies
echo Installing other dependencies...
venv\Scripts\pip install -r requirements.txt --quiet
echo.

echo ============================================================
echo  Setup complete!
echo  To run the app:   venv\Scripts\python app.py
echo  Or double-click:  run.bat
echo ============================================================
echo.
pause
