@echo off
echo ====================================================
echo  Starting AI Dataset Advisor Setup and Execution...
echo ====================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python 3.10 or higher and make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if venv folder exists
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment (venv)...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing/updating project dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] There was an issue installing some dependencies. Trying to proceed anyway...
)

REM Run the streamlit app
echo [INFO] Launching Streamlit dashboard...
streamlit run streamlit_app.py

echo ====================================================
echo  Session ended.
echo ====================================================
pause
