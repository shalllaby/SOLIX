# Chapter 7: DevOps, Local Deployment & Testing Suite

This chapter details the deployment architecture, environment configurations, and testing strategies implemented for the **SOLIX Platform (SOL Data Agent)**. To guarantee software reliability and seamless local deployment on developmental and presentation machines, the project follows structured DevOps workflows.

---

## 7.1 Local Deployment & System Requirements

The platform is designed to run locally as an autonomous desktop agent, making it self-contained and highly responsive.

### 7.1.1 System Requirements
*   **Operating System**: Windows 10/11 (specifically configured for CMD/PowerShell batch execution).
*   **Python Engine**: Python 3.10 or 3.11 (tested on Python 3.14 development preview).
*   **System Binaries**:
    *   **Tesseract OCR**: The binary must be installed on the local system (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`) to allow physical document reading.
    *   **Poppler**: Required for PyMuPDF PDF rendering utilities.

### 7.1.2 Automated Startup Script (`run_project.bat`)
To automate dependencies verification and application launching, a batch script `run_project.bat` is placed in the workspace root. The runner executes four key steps:
1.  **System Path Inspection**: Verifies if `python` is registered in the system environment PATH. If missing, it outputs an error and pauses execution.
2.  **Dependencies Verification**: Runs `pip install -r requirements.txt` silently to install or update missing packages.
3.  **UI Activation**: Automatically opens the default web browser to the hosting port `http://127.0.0.1:8000`.
4.  **Process Ingestion**: Starts the ASGI application server by running `python backend/main.py` and keeps the terminal window open for logs.

---

## 7.2 Environment Variable Configurations (`.env`)

Configuration parameters, API tokens, and secret keys are stored in a standard `.env` file in the workspace root. This isolates environment parameters from the source code.

| Environment Variable | Example Value | Description |
| :--- | :--- | :--- |
| `SECRET_KEY` | `b91a603957beab8d95...` | Secret key used to sign JWT session cookies securely. |
| `ALGORITHM` | `HS256` | JWT signature algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT access token lifetime in minutes. |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP email server hostname (Google Mail). |
| `SMTP_PORT` | `465` | SSL connection port for SMTP email dispatch. |
| `SMTP_USER` | `sol.data.agent@gmail.com` | Sender email address for OTP dispatches. |
| `SMTP_PASSWORD` | `gazo prwz nnau gcot` | Gmail App Password (bypassing 2FA locks). |
| `DATABASE_URL` | `sqlite:///./backend/data/sol.db` | Connection string for SQLite platform database. |
| `GOOGLE_CLIENT_ID` | `201884394525-tj7m...` | Client ID token for Google OAuth callbacks. |
| `GOOGLE_CLIENT_SECRET` | `GOCSPX-jCYZmZf...` | Client Secret token for Google OAuth callbacks. |
| `GITHUB_CLIENT_ID` | `Ov23liS7aIacmN6CfiAA` | Client ID token for GitHub OAuth callbacks. |
| `GITHUB_CLIENT_SECRET` | `51f3064c0b745395b...` | Client Secret token for GitHub OAuth callbacks. |
| `GROQ_API_KEY` | `gsk_js8xcosa1Coa...` | API key for high-speed Llama-3 model routing. |
| `ELEVENLABS_API_KEY` | `1` | API key for ElevenLabs Egyptian Arabic voice streaming. |
| `OPENROUTER_API_KEY` | `sk-or-v1-ef74b...` | DeepSeek LLM API token fallback. |
| `HF_TOKEN` | `hf_MPZcgvzdUXxu...` | Hugging Face token for local sentence embeddings. |
| `KAGGLE_USERNAME` | `mohamedshalaby11` | Kaggle credentials for remote training triggers. |
| `KAGGLE_KEY` | `KGAT_3c20f09026fc...` | Kaggle API authentication key. |

---

## 7.3 Testing Framework (Pytest Suite)

To ensure regressions do not slip into data processing or authentication pipelines, the platform includes a testing suite located in the `tests/` directory.

### 7.3.1 Pytest Architecture & Dependency Overrides
The test runner invokes **Pytest** with standard configurations. To prevent test runs from writing records to the live production database (`sol.db`), the testing suite uses **dependency injection overrides** in `tests/test_otp_auth.py`:

```python
# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency override
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
```

*   **Isolated Database**: By binding SQLAlchemy to `sqlite:///:memory:`, the database is created entirely in RAM for the duration of the test run and is automatically cleared afterward.
*   **Fixture Isolation**: The `@pytest.fixture(autouse=True)` block handles database creation (`Base.metadata.create_all`) before each test starts and teardown (`Base.metadata.drop_all`) after it completes.

### 7.3.2 Mocking Strategies & Sandbox Verification
*   **Email Dispatch Mocks**: Tests use Python's `unittest.mock.patch.object` to intercept the background SMTP worker (`email_worker.send_otp_email`). This allows tests to capture and inspect the generated 6-digit OTP code directly in memory without sending an actual email.
*   **OAuth Callback Mocks**: Since Google and GitHub authentication require redirects and external API requests, the test suite mocks `httpx.AsyncClient` post and get operations to return simulated authentication tokens and user profiles.
*   **Universal Loader Assertions**: Loader tests dynamically create temporary CSV, JSON, Excel, and Parquet files in a sandbox directory using Pytest's `tmp_path` fixture. These files are parsed using `DataLoaderFactory` and compared against the original Pandas DataFrame to verify parsing accuracy:

```python
def test_csv_loader(sample_dataframe, tmp_path):
    file_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    
    df = DataLoaderFactory.load_data(str(file_path))
    pd.testing.assert_frame_equal(df, sample_dataframe)
```

---

## 7.4 Expected Defense Committee Questions and Answers

### Question 1: How do you isolate tests so they don't corrupt the production database?
**Answer:** "We use FastAPI's dependency injection system. During testing, we override the default database connection dependency (`get_db`) with a temporary, in-memory SQLite connection (`sqlite:///:memory:`). This ensures all test transactions occur in memory and are discarded after the test finishes, keeping the production `sol.db` file untouched."

### Question 2: Why do you mock email dispatches instead of testing the SMTP server directly?
**Answer:** "Directly testing the SMTP server makes the tests slow and dependent on external network connectivity and mail server uptime. If the network drops or Google blocks the login request due to security policies, the test fails even if the code is correct. Mocking the `send_otp_email` function allows us to verify that the application generates a valid 6-digit code and calls the email dispatcher with the correct parameters, without sending an actual email."

### Question 3: How does the startup script ensure the system is ready to run?
**Answer:** "The batch script `run_project.bat` performs system checks before launching the application. It verifies that Python is installed and added to the system PATH. It then runs `pip install -r requirements.txt` to install any missing dependencies. Finally, it starts the FastAPI backend and launches the browser to `http://127.0.0.1:8000`, ensuring a smooth setup experience for developers or reviewers."
