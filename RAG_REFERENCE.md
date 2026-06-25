# SOL Data Agent — Comprehensive Technical Reference Manual (RAG Knowledge Base)

This document is the official, comprehensive technical reference manual and system documentation for the **SOL Data Agent** (نظام وكيل بيانات سول) — Enterprise Data Factory. It serves as the master source of truth for RAG indexing, search retrieval pipelines, AI chatbot assistants, and developer onboarding.

---

## 📌 1. Project Overview & Business Case

### 1.1 Project Identity
*   **Project Name:** SOL Data Agent (نظام وكيل بيانات سول) — Enterprise Data Factory.
*   **Aesthetic & UX Philosophy:** High-fidelity "Sci-Fi" dashboard employing Glassmorphism design principles (transparency, backdrop blurs, neon borders, and dark modes) alongside Egyptian Arabic voice-interactive capabilities.

### 1.2 Purpose and System Vision
SOL Data Agent is an autonomous, AI-powered data preparation, sanitization, and profiling engine. It bridges the gap between messy, real-world data and production-ready data assets by automating descriptive profiling, semantic type mapping, anomalous outlier detection, fuzzy typo correction, and predictive missing value imputation. 

### 1.3 Target Users & Business Goals
1.  **Enterprise Data Analysts:** Accelerate data cleaning before BI tool ingestion.
2.  **Machine Learning Engineers & Data Scientists:** Generate clean, high-integrity feature sets or high-fidelity synthetic datasets.
3.  **Non-Technical Business Owners:** Interact with structured data via a voice copilot to query, clean, and visualize datasets hands-free.

---

## 🛠️ 2. Technology Stack & Directory Structure

### 2.1 Core Technology Stack

```
   ┌──────────────────────────────────────────────────────────┐
   │                     PROGRAMMING LANGUAGES                │
   │            Python 3.11 (Backend) | Modern JavaScript     │
   └───────────────────────────┬──────────────────────────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────┐
   │                     FRAMEWORKS & UTILITIES               │
   │    FastAPI (ASGI, SSE) | Tailwind CSS | Jinja2 Templates │
   └───────────────────────────┬──────────────────────────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────┐
   │                 DATA SCIENCE & MODELING ENGINE           │
   │   Pandas | Polars | NumPy | Scikit-Learn | XGBoost | LightGBM│
   └───────────────────────────┬──────────────────────────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────┐
   │                 INFRASTRUCTURE & PERSISTENCE             │
   │ SQLite (aiosqlite ORM) | Qdrant Vector DB | ElevenLabs   │
   └──────────────────────────────────────────────────────────┘
```

*   **Backend Framework:** FastAPI (ASGI web framework) with `uvicorn` as the web server.
*   **Data Processing Engine:** Hybrid Pandas/Polars execution. Polars handles lazy, memory-efficient streaming, while Pandas manages active DataFrame modification in memory.
*   **Vector Database:** **Qdrant** (local storage format) for caching, indexing dataset descriptions, and semantic matching.
*   **Machine Learning Library:** Scikit-Learn (Random Forest classifiers/regressors) for missing value imputations.
*   **Third-Party AI Integrations:**
    *   **Cerebras & Groq APIs:** Ultra-low latency Llama-3/Llama-4 inference.
    *   **ElevenLabs API:** Conversational Egypt-colloquial text-to-speech synthesis.

### 2.2 Directory Map

```
run/
├── backend/                  # Python Backend Source
│   ├── auth.py               # JWT & OAuth handlers
│   ├── database.py           # SQLAlchemy setup
│   ├── main.py               # Main FastAPI entrypoint
│   ├── store.py              # In-memory Pandas DataFrames store
│   ├── models.py             # User and DB schemas
│   ├── tools/                # Subsystem Routers and Logic
│   │   ├── audit/            # Audit report generation
│   │   ├── automl/           # AutoML training & SSE pipeline
│   │   ├── copilot/          # ReAct Copilot routing & TTS
│   │   ├── dashboard/        # System statistics
│   │   ├── data_noise/       # Data corruptor & chaos engine
│   │   ├── dataset_advisor/  # Dataset Advisor, Qdrant & Kaggle Retrieval
│   │   ├── forms/            # SOL Forms builder & public endpoints
│   │   ├── ml_advisor/       # ML algorithm recommender
│   │   ├── narrator/         # Natural narration utilities
│   │   ├── ocr/              # Text/PDF OCR extraction
│   │   ├── semantic_mapper/  # Semantic schemas and mappings
│   │   ├── synthetic_data/   # Gaussian Copula/TVAE/CTGAN engines
│   │   └── viz_engine/       # Visualizations router
│   └── utils/
│       ├── ai_imputer.py     # Random Forest missing values imputer
│       └── policy_engine.py  # Data governance policy checkers
├── frontend/                 # Jinja2 Templates & Static Assets
│   ├── static/               # CSS, JS, Images, SVGs
│   └── templates/
│       ├── app/              # Dashboard/Workspace templates
│       └── public/           # Landing, Login, Registration templates
├── tests/                    # Pytest suite
├── requirements.txt          # Python dependencies
└── run_project.bat           # local startup script
```

---

## ⚡ 3. Detailed Features & Subsystems

### 3.1 Universal Loader
*   **Functional Details:** Supports CSV, Excel (`.xlsx`/`.xls`), JSON, XML, Parquet, Feather, HDF5, and ORC.
*   **Architecture:** The `DataLoaderFactory` dynamically resolves files, instantiates the correct parser, and registers the parsed DataFrame inside the thread-safe `_store` module map.

### 3.2 Smart Data Cleaner
*   **Cleaning Levels:**
    *   **Alpha:** General structural cleaning (dropping duplicate rows, stripping whitespace, resolving basic formatting anomalies).
    *   **Beta:** Outlier detection and removal using standard statistical thresholds (IQR/Z-score) and date/time standardization.
    *   **Gamma:** Advanced semantic cleaning, email/phone verification, and spelling correction (fuzzy mapping).
*   **Before/After Cell Diffs:** Frontend highlights changes in green (added/modified) or red (deleted/empty).

### 3.3 AI Imputer Engine (`AIImputer`)
*   **Algorithm:** Rather than using static methods (mean, median, mode), the engine trains custom Scikit-Learn models on-the-fly.
*   **Execution Flow:**
    1.  Splits target column into known and missing slices.
    2.  Encodes categorical features and scales numerical features from other columns as predictor variables ($X$).
    3.  Fits a `RandomForestRegressor` for numerical target columns or a `RandomForestClassifier` for categorical targets.
    4.  Imputes the missing records and updates the DataFrame.

### 3.4 SOL Voice Copilot
*   **Personality:** Dialogues in natural Egyptian Arabic with technical terminology.
*   **Dual-Agent System:**
    *   **Gatekeeper Agent (`refine_user_input`):** Categorizes input query as `CHAT` (social interaction) or `COMMAND` (data operation).
    *   **ReAct Execution Agent:** Generates Python/Pandas code. Runs code inside a sandbox, evaluates outcomes, and retries if execution fails or if no mutation occurred.
*   **Voice Pipeline:** Integrates with ElevenLabs TTS, capping responses to 25 words for fast voice streaming.

### 3.5 AutoML Studio
*   **Features:** Automates preprocessing, feature scaling, model selection (Random Forest, CatBoost, LightGBM, XGBoost), hyperparameter tuning, and cross-validation.
*   **Progress Streaming:** Utilizes Server-Sent Events (SSE) `text/event-stream` to push real-time training progress percentiles to the browser interface.
*   **Export:** Packages the trained pipeline artifacts (preprocessor `.pkl`, model binary, evaluation charts, and PDF summary) into a downloadable ZIP archive.

### 3.6 AI Dataset Advisor
*   **Intent Parser:** Understands unstructured queries (Arabic or English) and extracts key task types (e.g. classification), data modalities, and sizing boundaries.
*   **Live Retrieval:** Queries the Kaggle API for matching datasets. Returns to localized seeds database (such as ASTD Arabic Sentiment dataset) if offline.
*   **Qdrant Vector DB:** Encodes dataset titles and descriptions using a multilingual sentence-transformer model and stores vectors in Qdrant for semantic similarity searches.
*   **Hybrid Reranking:** Computes a composite score based on:
    *   Semantic content match (40%)
    *   Task fit (20%)
    *   Modality fit (15%)
    *   Language fit (15%)
    *   Row count boundary constraints (10%)

### 3.7 Visualizer, Auditor, & Chaos Engine
*   **Visualizer:** Renders dynamic data distributions, box plots, and correlation heatmaps.
*   **Auditor:** Maintains a detailed JSON audit log representing every dataframe modification, exportable as an audit PDF report.
*   **Chaos Engine:** Intentionally introduces errors, missing values, duplicates, and format corruption into datasets to test data pipeline resilience.

### 3.8 OCR & SOL Forms
*   **OCR:** Processes uploaded images and PDFs to extract tabular grids.
*   **SOL Forms:** A drag-and-drop form designer where public users can submit responses. Responses are appended directly to the database and can be exported as a dataset.

---

## 🗺️ 4. Data Flows & Execution Lifecycles

### 4.1 Zero-Token Ingestion Flow
To prevent API token exhaustion during file uploads, the initial profiling phase is performed entirely locally:

```
[Raw File Upload] ──► [DataLoaderFactory] ──► [Register in _store]
                                                    │
                                                    ▼
                                            [Polars Profiler]
                                            - Lazily scan dataset
                                            - Extract cardinality & nulls
                                            - Returns JSON schema
```

### 4.2 ReAct Sandbox Code Execution Loop

```
User Voice/Text Command ──► Gatekeeper ──► [Intent == COMMAND]
                                                │
                                                ▼
                                    [LLM Code Generation]
                                                │
                                                ▼
                                    [Sandbox Sandbox Run]
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼ (Success & Mutated)                                   ▼ (Fail or No Mutation)
        [Update _store Dataframe]                                [Feedback Error & Loop Cooldown]
                    │                                                       │
                    ▼                                                       ▼
        [Egypt-Arabic Synthesizer]                                     [Retry (Max 3)]
                    │
                    ▼
          [ElevenLabs Audio Stream]
```

---

## 🗄️ 5. Database Schema (SQLite)

The system uses SQLAlchemy ORM mapping to a local SQLite database (`sol.db` for core platform and `advisor.db` for the Dataset Advisor).

### 5.1 Platform Core Schema (`sol.db`)

#### Table: `users`
*   `id` (Integer, Primary Key)
*   `first_name` (String), `last_name` (String)
*   `username` (String, Unique), `email` (String, Unique)
*   `hashed_password` (String)
*   `status` (String, Default: `"active"`)
*   `job_title` (String), `organization` (String)
*   `created_at` (DateTime)

#### Table: `otp_sessions`
*   `id` (Integer, Primary Key)
*   `email` (String, Index)
*   `otp_hash` (String)
*   `expires_at` (DateTime)
*   `attempts` (Integer, Default: `0`)
*   `is_verified` (Boolean)
*   `created_at` (DateTime), `updated_at` (DateTime)

#### Table: `auth_logs`
*   `id` (Integer, Primary Key)
*   `ip_address` (String), `email` (String), `user_agent` (String)
*   `action` (String), `status` (String), `details` (String)
*   `timestamp` (DateTime)

#### Table: `forms`
*   `id` (Integer, Primary Key)
*   `title` (String), `description` (Text)
*   `questions` (JSON - Field schemas, options)
*   `created_at` (DateTime)

#### Table: `responses`
*   `id` (Integer, Primary Key)
*   `form_id` (Integer, Foreign Key to `forms.id`)
*   `answers` (JSON - Question-Answer mappings)
*   `timestamp` (DateTime)

---

### 5.2 Dataset Advisor Schema (`advisor.db`)

#### Table: `datasets`
*   `id` (String, Primary Key)
*   `kaggle_id` (String, Unique)
*   `title` (String), `description` (Text), `url` (String)
*   `row_count` (Integer), `column_count` (Integer)
*   `license` (String), `task_type` (String), `language` (String)
*   `tags` (JSON), `quality_score` (Float)
*   `created_at` (DateTime)

#### Table: `search_logs`
*   `id` (String, Primary Key)
*   `session_id` (String)
*   `query_text` (Text)
*   `detected_lang` (String)
*   `extracted_filters` (JSON)
*   `created_at` (DateTime)

#### Table: `recommendations`
*   `id` (String, Primary Key)
*   `log_id` (String, Foreign Key to `search_logs.id`)
*   `dataset_id` (String, Foreign Key to `datasets.id`)
*   `relevance_score` (Float)
*   `reasoning` (Text - AI generated rationale)
*   `created_at` (DateTime)

---

## 🔗 6. Core REST & SSE API Endpoints

### 6.1 Authentication
*   `POST /api/v1/auth/register` — Accepts user register fields. Triggers 6-digit OTP email dispatch.
*   `POST /api/v1/auth/verify-otp` — Verifies OTP sessions, sets secure cookies, returns JWT.
*   `POST /api/v1/auth/login` — Verifies credentials, returns JWT.
*   `GET /api/v1/auth/google/login` & `/api/v1/auth/github/login` — Initiate OAuth redirection.

### 6.2 AutoML Studio
*   `GET /api/automl/datasets` — Fetch list of dataset keys inside `_store`.
*   `POST /api/automl/upload-direct` — Direct multipart file upload. Returns parsed JSON schema.
*   `POST /api/automl/train` — Starts the pipeline. Returns an SSE progress stream:
    ```json
    event: progress
    data: {"event": "progress", "step": "Training LightGBM", "desc": "Tuning parameters...", "percent": 65}
    ```

### 6.3 Voice Copilot (SOL)
*   `POST /api/copilot/chat` — Submits message to the ReAct pipeline. Returns text output.
*   `GET /api/copilot/schema/{dataset_id}` — Resolves target schema from in-memory cache (Zero-Token).
*   `POST /api/copilot/tts` — Submits synthesized responses to ElevenLabs and streams the binary MP3 audio buffer.

### 6.4 Dataset Advisor
*   `POST /api/dataset-advisor/search` — Triggers intent parsing, semantic Qdrant searches, and Llama reasoning.
*   `GET /api/dataset-advisor/stats` — Returns count of local database datasets, search logs, and status of Qdrant connection.

---

## 🛡️ 7. Security, Sandboxing, & Safeguards

### 7.1 JWT & Cookie Security
The platform stores JWT parameters across two secure layers:
1.  `access_token` (HTTPOnly, SameSite=Strict): Secured cookie containing user identification, inaccessible via client-side scripts.
2.  `sol_auth_token` (SameSite=Strict): Accessible cookie utilized to hydrate frontend states.

### 7.2 Rate Limiting
The custom `SimpleRateLimiter` middleware tracks requests in-memory using client IP addresses. It blocks clients exceeding **30 requests per minute**, returning an HTTP `429 Too Many Requests` status code.

### 7.3 Code Execution Sandboxing
Pandas execution triggered by the Copilot's code agent runs in an isolated context:
*   Regex filter prevents import of dangerous packages (`os`, `sys`, `subprocess`, `shutil`, `importlib`).
*   Blocks dangerous built-in operations (`eval`, `exec`, `open`, `globals`, `locals`).
*   Uses localized dictionaries to sandbox mutating variables, protecting the host system from command injections.

### 7.4 Downstream API Safeguards
*   **ReAct Loop Sleep:** Injects `await asyncio.sleep(2)` between loops to avoid triggering Groq/Cerebras rate-limiting parameters.
*   **Fast Failures:** Configure `max_retries=0` for LLM calls during heavy loads to fail quickly instead of hanging and blocking FastAPI workers.
