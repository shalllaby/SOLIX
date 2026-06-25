Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Starting AI Dataset Advisor Setup and Execution... " -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# 1. Check Python
try {
    $pythonVersion = python --version
    Write-Host "[INFO] Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ and check 'Add Python to PATH'." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    exit
}

# 2. Check Virtual Environment
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "[INFO] Creating virtual environment (venv)..." -ForegroundColor Yellow
    python -m venv venv
}

# 3. Activate and Install dependencies
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

Write-Host "[INFO] Installing/updating project dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Run streamlit app
Write-Host "[INFO] Launching Streamlit dashboard..." -ForegroundColor Green
streamlit run streamlit_app.py

Read-Host "Press Enter to exit..."
