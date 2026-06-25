# Chapter 2: System Architecture and Directory Mapping

---

## Section 1: Software Architecture and System Design

### 1.1 Architectural Overview (Enterprise Data Factory Model)
The **SOL Data Agent** platform is built on an enterprise-grade, multi-tiered software architecture designed for high performance, modularity, and security. The platform implements a strict separation of concerns, decoupling the presentation layer (user interfaces) from the data processing engine, machine learning pipelines, and relational database systems.

The core of the backend is powered by **FastAPI**, an asynchronous, high-performance web framework for Python. FastAPI was selected due to its native support for asynchronous programming, auto-generated OpenAPI documentation, and low-latency response times. Real-time notifications and agent-thinking streams are pushed to the client using **Server-Sent Events (SSE)**, establishing a seamless, reactive interface.

The architecture comprises the following primary layers:
1. **Presentation Layer (Frontend)**: Developed using standard semantic HTML5, CSS3, and Vanilla JavaScript. This ensures that the platform is lightweight, highly performant, and free from heavy framework dependencies that could degrade render times.
2. **API and Controller Layer (FastAPI Routing)**: Manages API request routing, authentication, request validation, and rate limiting.
3. **Core Logic and AI Agent Layer (Engine Core)**: The operational heart of the system, comprising:
   * **Metadata Analyzer**: Extracts data structures and statistical summaries using the high-speed **Polars** library.
   * **AI Imputer & Cleaning Studio**: Handles automated data cleaning, missing value imputations, and outlier detection.
   * **Voice Copilot Engine**: Runs the conversational agent loops.
   * **Code Execution Sandbox**: Audits and executes generated Python scripts.
4. **Data Persistence Layer (Database & Storage)**: Uses SQLite database engines for local, low-latency session and project storage, alongside persistent file systems for uploaded data files.

---

### 1.2 Data Flow and Processing Lifecycle
Every user request, whether voice-based or text-based, goes through a highly regulated processing lifecycle:

```
[User Uploads File] ---> [Universal Loader Reads File Locally] ---> [Extract Dimensions/Schema]
                                                                               |
                                                                               v
[Egyptian Voice Welcomes User] <--- [Frontend Synthesizes Greeting] <--- [No-Token Metadata Response]
```

1. **Smart Ingestion (Zero-Token Metadata Extraction)**:
   * When a dataset (e.g., CSV or Excel) is uploaded via `POST /api/upload`, the backend reads it using the `DataLoaderFactory` inside `universal_loader.py`.
   * A schema summary, shape, and head preview are extracted locally. No data payloads or text contents are sent to remote LLMs during this phase.
   * The backend returns this schema summary. The frontend uses a localized greeting dictionary to generate a voice message (e.g., "Welcome! I have loaded your file 'sales.csv' containing 1,500 rows and 10 columns. What shall we clean first?"). This saves thousands of API tokens.
2. **User Command Processing (The Gatekeeper)**:
   * The user provides an input (voice or text), which is sent to `POST /api/chat`.
   * The input is parsed by the **Gatekeeper LLM Router**, running on the ultra-fast Cerebras Inference Engine.
   * The Gatekeeper categorizes the query into `CHAT` (general conversation) or `COMMAND` (data manipulation request) and outputs a structured JSON response:
     ```json
     {
       "intent": "COMMAND",
       "explanation": "The user wants to remove duplicate rows and fill missing values in 'age'."
     }
     ```
3. **The ReAct Execution Loop**:
   * If classified as a `COMMAND`, the request is sent to the **ReAct Agent**. The agent plans the solution, generates Python code using Pandas, and submits it to the Sandbox.
   * The Sandbox parses the code AST (Abstract Syntax Tree), ensuring no forbidden modules (e.g., `os`, `sys`, `socket`) or prohibited Pandas methods (e.g., raw `.fillna()`, which bypasses local auditing) are used.
   * If clean, the sandbox runs the code against a local copy of the DataFrame and captures `stdout` and any errors. If an error occurs, the ReAct loop automatically attempts to correct the code (up to 5 iterations).
4. **Voice Synthesis and Final Presentation**:
   * Upon successful execution, the modified DataFrame is saved, and an audit entry is logged.
   * The agent generates a brief, natural response in Egyptian Arabic (under 25 words) summarizing the outcome (e.g., "Done! I cleaned the duplicate records and imputed the missing values in the age column.").
   * This text is sent to the **ElevenLabs API** to synthesize high-quality voice audio, which is streamed back to the client along with the updated dataset preview and audit logs.

---
---

## Section 2: Directory Structure and Codebase Mapping

### 2.1 Project Directory Tree and File Roles
The project features a clean, highly modular layout. The following is the detailed folder and file breakdown of the SOLIX codebase:

```
SOLIX_PROJECT_ROOT/
│
├── backend/                        # FastAPI Web Server and API Gateways
│   ├── data/                       # Local SQLite databases (sol.db and advisor.db)
│   ├── middleware/                 # Security and request rate-limiting middlewares
│   ├── tools/                      # Feature-specific router endpoints and utilities
│   ├── workers/                    # Background workers for long-running processes
│   ├── auth.py                     # JWT token generation, verification, and cryptography
│   ├── auth_routes.py              # User signup, login, and OTP session endpoints
│   ├── database.py                 # SQLAlchemy engine configuration and sessions
│   ├── main.py                     # Core web application server and SSE endpoints
│   ├── migrate_db.py               # Database migration and table creation scripts
│   └── models.py                   # SQLAlchemy ORM models for database tables
│
├── core/                           # AI Logic, Data Imputation, and Execution Sandbox
│   ├── automl/                     # Automated Machine Learning algorithms and training
│   ├── copilot/                    # Voice Copilot logic (Sandbox, LLM clients, TTS, tools)
│   ├── analyzer.py                 # Statistical analysis and dataset quality assessment
│   ├── cleaner.py                  # Traditional and advanced data cleaning routines
│   └── strategist.py               # AI Recommendation and strategic advice generator
│
├── data_layer/                     # Data Loading and Integration Connectors
│   ├── connectors/                 # External database connectors (SQL, PostgreSQL)
│   └── loaders/                    # Universal file reading factory
│
├── frontend/                       # Client-Side Assets and Page Templates
│   ├── static/                     # Custom CSS stylesheets, JS scripts, images, and fonts
│   └── templates/                  # HTML Templates for views
│
└── requirements.txt                # Python package dependency manifest
```

---

### 2.2 Deep Dive: Core Code Module Analysis

#### 1. The Sandbox Engine (`core/copilot/sandbox.py`)
The sandbox provides secure execution of LLM-generated code. It checks the code AST before compiling and running it.
* **AST Audit (`_SecurityVisitor`)**:
  ```python
  class _SecurityVisitor(ast.NodeVisitor):
      def visit_Import(self, node: ast.Import) -> None:
          for alias in node.names:
              top_level = alias.name.split(".")[0]
              if top_level not in ALLOWED_IMPORTS:
                  raise SecurityError(f"Import of '{alias.name}' is forbidden.")
  ```
  Only modules in `ALLOWED_IMPORTS` (e.g., `pandas`, `numpy`, `math`, `scikit-learn`) are allowed.
* **Blocked Names**: Excludes strings like `"os"`, `"sys"`, `"subprocess"`, `"open"`, `"eval"`, `"exec"`, ensuring no command injection or file-system modifications can happen.
* **Forbidden Methods**: Prevents the use of raw `.fillna()`, `.dropna()`, or `KNNImputer`, forcing the LLM to use the system's audited wrapper functions (`smart_impute_column`, `clean_column_outliers`), which automatically write audit log metadata.
* **State Preservation**: The execution takes place in a copy of the dataframe, and audit logs are preserved in `df.attrs["audit_log"]` to ensure a complete audit trail.

#### 2. The Universal Data Loader (`data_layer/loaders/universal_loader.py`)
Utilizes the **Factory Pattern** to dynamically resolve the appropriate parser based on file extensions.
* **Supported Extensions**: Registers loaders for `.csv`, `.xlsx`, `.xls`, `.json`, `.parquet`, `.xml`, `.feather`, `.h5`, `.orc`, `.db`, and `.sqlite`.
* **Database Ingestion**: The `SQLLoader` dynamically creates an in-memory or temporary SQLite database connection and reads target tables into a Pandas DataFrame using SQLAlchemy.
* **Extensibility**: Developers can register custom loaders at runtime using `DataLoaderFactory.register_loader(extension, loader_class)`.

#### 3. Database ORM Schemas (`backend/models.py`)
Defines the relational schema of the platform. Key tables include:
* `User`: Standard user credentials, roles, and profiles.
* `OTPSession`: Verifies user logins securely via temporary tokens.
* `Project`: Groups datasets, cleaning states, and execution histories.
* `Task` & `TaskRun`: Tracks individual data tasks and automated background script executions.
* `TokenUsageRecord`: Records APIs token consumptions (Cerebras, OpenAI, ElevenLabs) for usage billing.
* `Feedback`: Collects user feedback on agent execution quality.
