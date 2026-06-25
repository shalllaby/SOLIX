@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

title SOL Data Agent Runner
color 0A

echo ====================================================================
echo                   SOL DATA AGENT - RUNNER
echo ====================================================================
echo.
echo  [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Python is not installed or not added to your system PATH.
    echo  Please install Python 3.10+ and ensure 'Add Python to PATH' is checked.
    echo.
    pause
    exit /b 1
)

echo  [2/3] Checking dependencies (FastAPI, Pandas, etc.)...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo  [WARNING] Could not verify/update dependencies via pip. Attempting to start server anyway...
) else (
    echo  [OK] Dependencies verified.
)
echo.

echo  [3/3] Launching Web Browser at http://127.0.0.1:8000...
start http://127.0.0.1:8000
echo.

echo  --------------------------------------------------------------------
echo  Starting Application Server...
echo  (Keep this terminal window open while using the application)
echo  --------------------------------------------------------------------
echo.
python backend/main.py
if %errorlevel% neq 0 (
    echo.
    echo  [WARNING] The server closed or failed to start.
    echo  Please check the logs above.
    pause
)
