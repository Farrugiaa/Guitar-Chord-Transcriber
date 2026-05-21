@echo off
if not exist venv (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
venv\Scripts\python app.py
