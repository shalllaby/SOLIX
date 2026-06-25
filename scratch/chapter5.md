# Chapter 5: Backend Development & API Core Specification

This chapter provides a detailed, comprehensive deep dive into the backend development of the **SOLIX Platform (SOL Data Agent)**. The backend serves as the core orchestration layer of the system—managing the lifecycle of uploaded datasets, handling secure user authentication, checking execution safety, integrating AI engines, and compiling PDF cleaning audits.

The backend is built using **FastAPI**, a modern, high-performance web framework for building APIs with Python. It is chosen for its native asynchronous capabilities, automatic API documentation (via Swagger UI and ReDoc), and structural validation powered by **Pydantic**.

---

## 5.1 FastAPI Core Architecture & Configurations

The backend entry point is configured inside `backend/main.py`. Upon application startup, the database engine initializes connection parameters, migrates schemas, mounts static assets, and registers a suite of feature-specific routers.

```
+---------------------------------------------------------------------------------+
|                                 FASTAPI CLIENT                                 |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       SimpleRateLimiter Middleware (Auth API)                    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         SessionMiddleware (OAuth / Cookies)                     |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                        CORSMiddleware (Origins: * Wildcard)                      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +------------------+            +------------------+
|   Auth Routes    |           |   Data Cleaning  |            |   Voice Copilot  |
|  (OTP, JWT, OAuth) |         |  (Upload/Parquet)|            | (Sandbox/ Eleven)|
+------------------+           +------------------+            +------------------+
```

### 5.1.1 CORSMiddleware & Session Configurations
To allow the frontend templates and external API integrations to interact securely with the API, CORS is configured to allow wildcard origins, credentials, and custom headers:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Additionally, `SessionMiddleware` is mounted to provide secure, encrypted session states required for OAuth callbacks:

```python
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "b91a603957beab8d956f2f9f98f6d89bdfad741df747372cf91c0e358b68832a")
)
```

---

## 5.2 SimpleRateLimiter Middleware

To prevent credential stuffing, brute-force OTP attempts, and denial-of-service (DoS) attacks on the authentication endpoints, the application implements a custom in-memory rate limiting middleware defined in `backend/middleware/rate_limit.py`.

### 5.2.1 Mechanics of the Rate Limiter
1. **Targeting**: The rate limiter selectively applies to authentication routes beginning with `/api/v1/auth` to prevent interfering with static page delivery or frontend page loads.
2. **Identification**: Clients are identified by their IP address, resolved through the `x-test-ip` header (for testing environments) or the standard FastAPI `request.client.host` parameter.
3. **Limiting**: Default limits are configured at **30 requests per 60 seconds** per IP address.
4. **Implementation details**: It maintains an in-memory dictionary `history: Dict[str, List[float]]` storing UNIX timestamps of client hits. During each request:
    * Timestamps older than 60 seconds are purged.
    * If the count of remaining timestamps exceeds the threshold, the request is blocked and returns an `HTTP 429 Too Many Requests` JSON response.
    * Otherwise, the current timestamp is appended, and the request proceeds to the router.
5. **Testing bypass**: In automated test suites (detected via `pytest` module presence or the `TESTING=1` environment variable), the rate limiter is bypassed unless the IP is explicitly set to `"ratelimit-ip"`.

---

## 5.3 Authentication, Authorization, and Security Flows

The platform implements a multi-tiered security system combining standard username/password authentication, email verification via Secure One-Time Passwords (OTP), and JSON Web Token (JWT) session management.

### 5.3.1 Secure User Registration and Password Strength Validation
The user registration model (`RegisterSchema`) enforces strict security validation at the Pydantic parser layer using `@field_validator`:
*   **Minimum Length**: Passwords must be at least 8 characters long.
*   **Required Characters**: Must contain at least one digit and one special character (e.g., `!@#$%^&*`).

When a registration request passes validation, a unique username is automatically compiled from the user's first and last names (e.g., `first_last`). To prevent username collisions, a loop queries the database and appends numeric increments (e.g., `first_last_1`) until an available slot is found:

```python
base_username = re.sub(r"[^a-zA-Z0-9_]", "", (payload.first_name + "_" + payload.last_name).lower())
username = base_username
counter = 1
while db.query(User).filter(User.username == username).first():
    username = f"{base_username}_{counter}"
    counter += 1
```

### 5.3.2 OTP Generation, Hashing, and Lockout Checks
When an account is created, it remains in a `pending_verification` status. The system issues a 6-digit numeric OTP and queues an email via `email_worker.send_otp_email`.

```
                    +------------------------------------+
                    | User Registration (Status: Pending)|
                    +------------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    | Generate 6-Digit Numeric OTP Code  |
                    +------------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    | SHA-256 Hash of OTP Saved to DB    |
                    +------------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   Send Plaintext OTP to User Email |
                    +------------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
          [Incorrect Attempt]                   [Correct Attempt]
                    |                                   |
           Attempts Incremented                 User Status: Active
                    |                                   |
           Lockout if Attempts >= 5                     |
                    v                                   v
             Access Blocked                     Generate JWT Cookie
```

To protect OTPs from database-theft vectors:
*   **SHA-256 Hashing**: OTPs are hashed using SHA-256 before being stored in the `otp_sessions` table. This prevents database administrators or attackers with read access from reading active codes.
*   **Lockout Policy**: A maximum of 5 validation attempts are permitted (`OTP_MAX_ATTEMPTS = 5`). If reached, the user is locked out until they request a new OTP.
*   **Expiration**: OTPs expire exactly 5 minutes after issuance.
*   **Replay Protection**: The `is_verified` flag prevents verified OTPs from being re-used.

### 5.3.3 JWT Session Tokens & Cookie Policies
Authentication is persisted using JSON Web Tokens (JWT) using the `HS256` signature algorithm. 

To ensure compatibility across different frontend components (e.g., AJAX fetch requests vs standard server-rendered HTML page requests), the login route sets two separate cookies:
1.  `access_token`: A secure, `HttpOnly` cookie. This prevents client-side JavaScript from reading the token, securing it against Cross-Site Scripting (XSS) attacks.
2.  `sol_auth_token`: A standard cookie accessible via JavaScript (`httponly=False`). This allows client-side scripts to verify if the user is authenticated without having to ping the server.

Both cookies use the `samesite="strict"` policy to defend against Cross-Site Request Forgery (CSRF) vulnerabilities.

### 5.3.4 Google and GitHub OAuth Integrations
For single-sign-on (SSO), Google and GitHub OAuth 2.0 flows are integrated. 
*   **Callback Normalization**: Google and GitHub OAuth require exact callback URLs. The backend normalizes the redirect URLs dynamically: if running on local subnets or `127.0.0.1`, it rewrites them to `localhost:8000` to prevent redirect mismatch exceptions.
*   **Auto-Registration**: Upon callback, the backend exchanges the authorization code for an access token, fetches the user's profile and verified email address, and creates a user account automatically with `active` status.

---

## 5.4 REST API Endpoints Specification

The backend exposes a structured API catalog grouped by functionality.

| HTTP Method | Endpoint | Authenticated | Description |
| :--- | :--- | :--- | :--- |
| **POST** | `/api/v1/auth/register` | No | Creates a new user in `pending_verification` status and sends an OTP. |
| **POST** | `/api/v1/auth/verify-otp` | No | Verifies the OTP, activates the user, and sets JWT cookies. |
| **POST** | `/api/v1/auth/resend-otp` | No | Re-sends a new OTP after a 60-second cooldown period. |
| **POST** | `/api/v1/auth/login` | No | Authenticates credentials and issues JWT cookies. |
| **POST** | `/api/v1/auth/logout` | Yes | Clears the authentication cookies. |
| **GET** | `/api/v1/auth/google/login` | No | Redirects the user to the Google Consent screen. |
| **GET** | `/api/v1/auth/google/callback` | No | Processes Google token exchange and logs in the user. |
| **POST** | `/api/upload` | Yes | Receives chunked file uploads and saves raw files as Parquet. |
| **POST** | `/api/clean` | Yes | Triggers an AI dataset cleaning job. |
| **POST** | `/api/corrupt` | Yes | Injects data noise/anomalies for training or evaluation. |
| **POST** | `/api/undo` | Yes | Reverts the last dataset cleaning operation. |
| **POST** | `/api/redo` | Yes | Re-applies the last reverted dataset cleaning operation. |

### 5.4.1 Undo and Redo Mechanics
The backend tracks data cleaning edits inside an active session registry (`_store`).
*   When a clean or corrupt action occurs, a snapshot of the dataframe is saved to disk under `temp_snapshots/{dataset_id}_v{version}.parquet`.
*   An array of active version paths is maintained in memory.
*   `/api/undo` decrements the active version index, reads the target Parquet file from disk via Polars, and registers it as the current active dataset.
*   `/api/redo` increments the version index and re-loads the corresponding Parquet file, providing instant, memory-safe version tracking.

---

## 5.5 CleaningStudioPDFReportGenerator

After clean processing, users can export a PDF audit summary. The PDF is generated dynamically using **ReportLab** and **Matplotlib** inside `backend/utils/cleaning_studio_pdf_generator.py`.

```
                  +-----------------------------------------+
                  |    Initiate PDF Generation Request      |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  |  Load & Register Multilingual TrueType  |
                  |     Fonts (Arial / Calibri / Tahoma)    |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  | Generate Missing Value Chart with       |
                  |  Matplotlib (Before vs After Clean)     |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  | Apply arabic_reshaper & python-bidi     |
                  |    for RTL Text Layout Processing       |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  | Compile Sections into ReportLab Flow   |
                  | (Tables, Charts, Audit Logs, Metadata)  |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  |   Build Document & Return Bytes Buffer  |
                  +-----------------------------------------+
```

### 5.5.1 Multilingual Support (Arabic & English RTL Rendering)
ReportLab does not support Arabic text out of the box because Arabic letters change shape depending on their position in a word and are read from right to left. The generator handles this via a custom text processing pipeline:
1.  **Font Registration**: The generator checks standard Windows system font directories to register `arial.ttf` and `arialbd.ttf` (or Calibri/Tahoma). If found, it registers them under the alias `MultilingualArial`.
2.  **Arabic Reshaping**: It uses `arabic_reshaper` to merge Arabic letters into their correct cursive glyph representations.
3.  **Bidirectional Layout**: It uses `bidi.algorithm.get_display` to invert the character rendering sequence from right-to-left.
4.  **Alignment Adjustments**: When generating reports in Arabic, text styles are automatically updated with right alignment (`alignment = 2`).

### 5.5.2 Matplotlib Visual Embedding
To make the report professional, the generator builds a horizontal comparison bar chart using Matplotlib:
*   It visualizes the count of missing values (nulls) before (Red bar) vs after (Green bar) the cleaning process.
*   The chart is saved directly to an in-memory bytes buffer (`io.BytesIO()`) as a high-density PNG to avoid writing temporary files.
*   The buffer is wrapped inside ReportLab's Flowable `Image` container and inserted into the PDF document flow.

### 5.5.3 Compiled Sections
The final compiled PDF report consists of five distinct sections:
1.  **Technology Stack & Execution Context**: Lists the execution platform (Local Fallback vs Remote Kaggle Instance), strategy, user goals, and core engineering libraries.
2.  **Executive Cleaning Statistics**: A detailed table comparing initial rows vs cleaned rows, missing cells, and the final *Truth Confidence Score*.
3.  **Enterprise Quality Guardrails**: Outlines blocked actions, warnings, and successfully permitted operations.
4.  **Performance & Schema Diagnostics**: Records execution runtime, peak memory allocated, and schema mutations (e.g., data types mutated, dropped, or preserved).
5.  **Detailed Anomalies Audit Log**: A granular table listing every single anomaly resolved, complete with column names, detected issues, and resolution methods.

---

## 5.6 Expected Defense Committee Questions and Answers

### Question 1: Why did you choose FastAPI over Flask or Django?
**Answer:** "FastAPI was selected because it is built on top of Starlette and Uvicorn, which support asynchronous operations (async/await) out of the box. This makes it significantly faster than Flask and Django for handling simultaneous I/O-bound requests like streaming large dataset uploads or communicating with remote AI engines. Additionally, FastAPI provides automatic request/response validation using Pydantic schemas, reducing security vulnerabilities such as SQL injections or buffer overflows, and automatically hosts interactive Swagger documentation."

### Question 2: How does your SimpleRateLimiter middleware protect the server?
**Answer:** "The middleware intercepts all HTTP requests hitting the `/api/v1/auth` endpoints. It extracts the client IP address and tracks the request history in memory over a sliding window of 60 seconds. If an IP address attempts to send more than 30 requests in a single minute, the limiter halts execution early and returns a 429 status code. This prevents brute-force attempts on user passwords or verification OTPs, and reduces server CPU overhead."

### Question 3: How do you secure OTPs in the database against security breaches?
**Answer:** "We never store OTPs in plaintext. Instead, when a 6-digit verification code is generated, the backend computes its SHA-256 hash and saves the hash in the `otp_sessions` table. During verification, the submitted code is hashed and compared to the database record. If an attacker gains unauthorized access to our database, they will only see irreversible SHA-256 hashes, protecting active verification sessions. Additionally, OTPs are secured with a 5-minute expiration time and a lockout policy that freezes the session after 5 failed attempts."
