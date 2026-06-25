# Chapter 6: Database Design & Schema Specification

This chapter provides a detailed, comprehensive blueprint of the database design for the **SOLIX Platform (SOL Data Agent)**. To achieve maximum portability, low-latency disk operations, and simplified installation for local AI deployment, the platform utilizes an embedded, relational database engine.

The system partitions its storage into two distinct SQLite databases located in `backend/data/`:
1.  **`sol.db`**: The Core Platform Database, responsible for user registration, audit logs, authentication records, job track files, public templates, and forms.
2.  **`advisor.db`**: The Dataset Advisor Database, responsible for maintaining cataloged dataset seeds, query history, and AI recommendation rationale.

Additionally, a local instance of **Qdrant** is integrated as a vector database for semantic similarity operations.

---

## 6.1 Relational Database Architecture Overview

SQLite is chosen as the primary relational database for the following architectural reasons:
*   **Zero Configuration**: There is no database server to install, configure, or secure, which simplifies deployment.
*   **Single-File Storage**: The entire database is contained in a single disk file, making backups and migrations as simple as copying the file.
*   **Embedded Execution**: Database queries run inside the same process as the FastAPI backend, eliminating network round-trip overhead.
*   **Full SQL Support**: Supports transactions, complex joins, primary keys, and foreign keys.

To manage database operations, the backend utilizes **SQLAlchemy**, a popular Python Object-Relational Mapper (ORM). SQLAlchemy decouples database operations from SQL dialect specifics, allowing developers to interact with models as standard Python objects.

---

## 6.2 Platform Core Database Schema (`sol.db`)

The primary database `sol.db` contains 12 tables that manage the platform's core states.

```
                    +---------------+
                    |     users     |
                    +---------------+
                      |           |
         +------------+           +------------+
         |                                     |
         v                                     v
+------------------+                  +------------------+
|   otp_sessions   |                  |    job_records   |
+------------------+                  +------------------+
                                               |
                                               v
+------------------+                  +------------------+
|      tasks       | <-------------- |    task_runs     |
+------------------+                  +------------------+
         ^
         |
+------------------+
|     projects     |
+------------------+
```

### 6.2.1 Table: `users`
Stores user profile information, encrypted password credentials, and account activation states.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key, Index | Unique identifier for each user. |
| `first_name` | VARCHAR | Nullable, Index | First name of the user. |
| `last_name` | VARCHAR | Nullable, Index | Last name of the user. |
| `username` | VARCHAR | Unique, Index | Unique handle (automatically generated). |
| `email` | VARCHAR | Unique, Index, Not Null | Email address (used for logging in). |
| `hashed_password` | VARCHAR | Not Null | Hashed password string (bcrypt). |
| `status` | VARCHAR | Default: `"active"` | Registration status: `"pending_verification"` or `"active"`. |
| `job_title` | VARCHAR | Nullable | Job title of the user. |
| `organization` | VARCHAR | Nullable | Organization of the user. |
| `avatar_url` | VARCHAR | Nullable | Link to user avatar image file. |
| `created_at` | DATETIME | Server Default: NOW | Timestamp when the account was created. |

### 6.2.2 Table: `otp_sessions`
Manages verification sessions, hashing verification codes to verify emails securely.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key, Index | Unique session identifier. |
| `email` | VARCHAR | Index, Not Null | Target email address to verify. |
| `otp_hash` | VARCHAR | Not Null | SHA-256 hash of the 6-digit verification code. |
| `expires_at` | DATETIME | Not Null | Expiration timestamp (set to +5 minutes). |
| `attempts` | INTEGER | Default: `0` | Number of failed verification attempts. |
| `is_verified` | BOOLEAN | Default: `False` | True if the code has been verified. |
| `created_at` | DATETIME | Default: UTC NOW | Timestamp when the code was created. |
| `updated_at` | DATETIME | Default: UTC NOW | Timestamp when the record was last updated. |

### 6.2.3 Table: `auth_logs`
An audit trail table recording authentication events, failed logins, and password resets.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key, Index | Unique log identifier. |
| `ip_address` | VARCHAR | Nullable | IP address of the client request. |
| `email` | VARCHAR | Index, Nullable | Email address associated with the event. |
| `user_agent` | VARCHAR | Nullable | Web browser identifier of the client. |
| `action` | VARCHAR | Not Null | Event action: `"register"`, `"login_success"`, `"otp_failed"`. |
| `status` | VARCHAR | Default: `"info"` | Status level: `"success"`, `"failed"`, `"info"`. |
| `details` | VARCHAR | Nullable | Descriptive explanation of the event. |
| `timestamp` | DATETIME | Default: UTC NOW | Date and time of the event. |

### 6.2.4 Table: `job_records`
Tracks data pipeline execution history, dataset metrics, and accuracy rates.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key, Index | Unique database identifier. |
| `task_id` | VARCHAR | Unique, Index, Not Null | Unique task run token (UUID format). |
| `user_id` | INTEGER | ForeignKey(`users.id`) | Links to the user who ran the job. |
| `task_type` | VARCHAR | Not Null | Job type: `"cleaning"`, `"synthetic"`, `"ocr"`, `"automl"`. |
| `filename` | VARCHAR | Nullable | Original uploaded file name. |
| `file_size_bytes` | INTEGER | Default: `0` | Size of the file in bytes. |
| `row_count` | INTEGER | Nullable | Total rows in the processed dataset. |
| `col_count` | INTEGER | Nullable | Total columns in the processed dataset. |
| `strategy` | VARCHAR | Nullable | Cleaning level applied: `"Alpha"`, `"Beta"`, `"Gamma"`. |
| `status` | VARCHAR | Default: `"pending"` | Job status: `"pending"`, `"processing"`, `"completed"`, `"failed"`. |
| `accuracy_rate` | FLOAT | Nullable | Imputation quality or classification accuracy score. |
| `error_message` | TEXT | Nullable | Execution error trace if the job fails. |
| `created_at` | DATETIME | Default: UTC NOW | Date and time when the job was requested. |
| `updated_at` | DATETIME | Default: UTC NOW | Date and time when the job finished. |

### 6.2.5 Table: `projects`
Organizes work into projects for workspaces.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | Primary Key, Index | Unique identifier (UUID). |
| `title` | VARCHAR | Not Null | User-facing project title. |
| `description` | TEXT | Nullable | Summary of the project. |
| `created_at` | DATETIME | Default: UTC NOW | Date and time when the project was created. |
| `updated_at` | DATETIME | Default: UTC NOW | Date and time when the project was last updated. |

### 6.2.6 Table: `tasks`
Tracks tasks created within a project.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | Primary Key, Index | Unique identifier (UUID). |
| `project_id` | VARCHAR | ForeignKey(`projects.id`) | Project associated with this task. |
| `name` | VARCHAR | Not Null | Name of the task. |
| `status` | VARCHAR | Default: `"PENDING"` | Status: `"PENDING"`, `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. |
| `progress_percentage` | INTEGER | Default: `0` | Completion percentage (0 to 100). |
| `state_data` | JSON | Nullable | Serialized state information. |
| `created_by` | INTEGER | ForeignKey(`users.id`) | User who created the task. |
| `assigned_worker_id`| VARCHAR | Nullable | Worker process identifier. |
| `version_id` | INTEGER | Default: `1` | Incrementing version number. |
| `created_at` | DATETIME | Default: UTC NOW | Date and time when the task was created. |
| `updated_at` | DATETIME | Default: UTC NOW | Date and time when the task was last updated. |

### 6.2.7 Table: `task_runs`
Tracks history for individual tasks.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | Primary Key, Index | Unique identifier (UUID). |
| `task_id` | VARCHAR | ForeignKey(`tasks.id`) | Associated task identifier. |
| `started_at` | DATETIME | Default: UTC NOW | Date and time when the run started. |
| `finished_at` | DATETIME | Nullable | Date and time when the run finished. |
| `error_log` | TEXT | Nullable | System error trace if the run fails. |
| `result_metadata` | JSON | Nullable | Output metrics and results. |

### 6.2.8 Tables: `forms` & `responses`
Used by the dynamic form-building tool.

#### Table: `forms`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key, Index | Unique form identifier. |
| `title` | VARCHAR | Index, Not Null | Form title. |
| `description` | TEXT | Nullable | Form description. |
| `questions` | JSON | Not Null | Questions list schema. |
| `created_at` | DATETIME | Server Default: NOW | Creation timestamp. |

#### Table: `responses`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key, Index | Unique response identifier. |
| `form_id` | INTEGER | ForeignKey(`forms.id`) | Associated form identifier. |
| `answers` | JSON | Not Null | User submitted answers. |
| `timestamp` | DATETIME | Server Default: NOW | Submission timestamp. |

### 6.2.9 Tables: `notifications`, `token_usage_records` & `feedbacks`
*   **`notifications`**: Stores system updates, successes, or failures for users. Linked via `user_id` to `users.id`.
*   **`token_usage_records`**: Logs API token usage for billing and performance audits. Stores model names, token counts, module sources, and timestamps.
*   **`feedbacks`**: Stores feedback submitted via landing page forms. Stores names, emails, phones, messages, and timestamps.

---

## 6.3 Dataset Advisor Database Schema (`advisor.db`)

The Dataset Advisor uses a separate database file `advisor.db` to isolate search history and recommendation logs from the core database.

### 6.3.1 Table: `datasets`
Acts as a local cache for metadata retrieved from the Kaggle API.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | Primary Key | Unique catalog key. |
| `kaggle_id` | VARCHAR | Unique, Not Null | Kaggle path (e.g. `owner/dataset-name`). |
| `title` | VARCHAR | Not Null | Dataset title. |
| `description` | TEXT | Nullable | Markdown dataset description. |
| `url` | VARCHAR | Nullable | Link to the download source. |
| `row_count` | INTEGER | Nullable | Number of rows in the dataset. |
| `column_count` | INTEGER | Nullable | Number of columns in the dataset. |
| `license` | VARCHAR | Nullable | Dataset license type (e.g. CC0, MIT). |
| `task_type` | VARCHAR | Nullable | Task type (e.g. classification, regression). |
| `language` | VARCHAR | Nullable | Primary language of the dataset. |
| `tags` | JSON | Nullable | List of tags. |
| `quality_score` | FLOAT | Nullable | Kaggle usability rating. |
| `created_at` | DATETIME | Default: UTC NOW | Timestamp when cached. |

### 6.3.2 Table: `search_logs`
Logs user searches to help improve recommendations.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | Primary Key | Unique log identifier. |
| `session_id` | VARCHAR | Index | Active user session identifier. |
| `query_text` | TEXT | Not Null | Search query entered by the user. |
| `detected_lang` | VARCHAR | Nullable | Language code of the query. |
| `extracted_filters` | JSON | Nullable | Filters extracted by the parser. |
| `created_at` | DATETIME | Default: UTC NOW | Search timestamp. |

### 6.3.3 Table: `recommendations`
Stores generated dataset recommendations for later retrieval.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | Primary Key | Unique recommendation identifier. |
| `log_id` | VARCHAR | ForeignKey(`search_logs.id`)| Links to the search query. |
| `dataset_id` | VARCHAR | ForeignKey(`datasets.id`) | Links to the recommended dataset. |
| `relevance_score` | FLOAT | Not Null | Computed match score. |
| `reasoning` | TEXT | Not Null | Explanations generated by the AI model. |
| `created_at` | DATETIME | Default: UTC NOW | Timestamp when created. |

---

## 6.4 Schema Migrations and Maintenance

To handle updates without losing existing data, the application uses a custom schema migration checker in `backend/migrate_db.py`.

*   **Runtime Verification**: When the server starts up, it connects to the database via Python's standard `sqlite3` driver.
*   **PRAGMA Inspection**: It runs `PRAGMA table_info(users)` and `PRAGMA table_info(auth_logs)` to inspect the database structure.
*   **In-Place Alternation**: If new columns (such as `username`, `status`, or `job_title`) are missing, it executes `ALTER TABLE users ADD COLUMN` statements.
*   This approach ensures schema updates are applied seamlessly on the user's local system without requiring manual database migrations.

---

## 6.5 Vector Search Database (Qdrant DB Setup)

To support semantic searches inside the Dataset Advisor, the platform integrates **Qdrant**, a high-performance vector database.
*   **Embedding Generation**: Dataset titles and descriptions are processed using a local transformer model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) to generate 384-dimensional dense vectors.
*   **Index Creation**: Vectors are indexed in Qdrant collections using Cosine Similarity metrics.
*   **Search Ingestion**: When a user submits a search query, it is converted into a vector and matched against the Qdrant collection to retrieve the most semantically relevant datasets.

---

## 6.6 Expected Defense Committee Questions and Answers

### Question 1: Why did you choose SQLite over PostgreSQL or MySQL for the final product?
**Answer:** "SOLIX is designed as an autonomous desktop agent that users can run locally. Using PostgreSQL or MySQL would require the user to install and configure an external database server, which adds complexity. SQLite runs in-process, saves all database tables in a single file, and requires no configuration. This ensures that the agent works immediately out-of-the-box, while still providing full SQL transactions, joins, and indexing."

### Question 2: SQLite does not strictly enforce foreign key constraints by default. How do you handle this?
**Answer:** "SQLite requires foreign key validation to be enabled per-connection. In SQLAlchemy, we configure this by listening to connection pool events. When a connection is opened, we run the SQL command `PRAGMA foreign_keys = ON;` to ensure SQLite enforces database referential integrity. Additionally, we define foreign key cascade deletes (e.g., `ondelete="CASCADE"` on `task_runs`) at the database schema level."

### Question 3: Why did you split the database into sol.db and advisor.db?
**Answer:** "This separates core platform tables from application-specific caches. `sol.db` contains user profile information, security audit logs, and forms. `advisor.db` acts as a local cache for dataset searches and AI recommendation logs. Splitting them keeps the main user database clean and lightweight, and allows the advisor cache to be cleared or rebuilt without affecting user accounts or platform audit trails."
