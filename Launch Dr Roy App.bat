@echo off
REM ============================================================
REM   DR. ROY DATA TRAINING & REPORTING - one-click launcher
REM   Double-click this file to start the app.
REM ============================================================
cd /d "%~dp0"
echo Starting Dr. Roy App...
"%~dp0venv\Scripts\pythonw.exe" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo The app could not start. Running diagnostics...
    "%~dp0venv\Scripts\python.exe" "%~dp0check_setup.py"
    pause
)
