# Database Multi-Tenancy & Data Isolation Remediation Plan

This document outlines the architectural roadmap for securing the application against **Data Cross-Contamination** by transitioning from shared/global states to a strictly isolated, multi-tenant database design.

---

## 1. The "Blast Radius" Audit (Exhaustive Entity Classification)

We have audited 100% of the active database models (across the main database `sol.db` and the advisor database `advisor.db`) along with system-level configuration singletons. Below is the strict categorization:

### `[NATURALLY GLOBAL]`
*Tables containing public, catalog, or system-wide static data:*
1. **`datasets`** (`advisor.db`): Kaggle and system-seeded datasets catalog for the advisor tool. Shared globally.
2. **`feedbacks`** (`sol.db`): Global website contact/support forms sent to administrators.
3. **`auth_logs`** (`sol.db`): Security auditing, failed logins, and access logging monitored globally by site admins.

### `[ALREADY ISOLATED]`
*Tables that correctly implement user-scoping via direct foreign key mapping:*
1. **`users`** (`sol.db`): The root identity/tenant record.
2. **`projects`** (`sol.db`): Contains `user_id` referencing `users.id` with `ondelete="CASCADE"`.
3. **`forms`** (`sol.db`): Contains `user_id` referencing `users.id` with `ondelete="CASCADE"`.
4. **`notifications`** (`sol.db`): Contains `user_id` referencing `users.id`.
5. **`job_records`** (`sol.db`): Contains `user_id` referencing `users.id`.
6. **`tasks`** (`sol.db`): Scoped via `created_by` referencing `users.id`.

### `[CRITICAL HAZARDS]`
*Entities currently lacking user-scoping that allow data cross-contamination:*
1. **`Settings / Credentials`** (System Config): Currently stored globally in the host `.env` file. All users read/write the same Kaggle/Groq credentials. **Remediation**: Create a `user_settings` table.
2. **`Dashboard Preferences`** (System Layout): Currently non-existent in the schema; dashboard configurations/preferences are shared globally. **Remediation**: Create a `dashboard_preferences` table.
3. **`responses`** (`sol.db`): Form submissions. Currently linked only to `form_id`. Lacks owner/submitter mapping. **Remediation**: Add `user_id` to allow user-scoped data retrieval.
4. **`task_runs`** (`sol.db`): Job executions/logs. Currently linked only to `task_id`. **Remediation**: Add `user_id` for direct relationship and ownership verification.
5. **`token_usage_records`** (`sol.db`): Tracked LLM usage. Currently lacks `user_id`, preventing tenant-level token capping. **Remediation**: Add `user_id` column.
6. **`search_logs`** & **`recommendations`** (`advisor.db`): Dataset advisor queries. Scoped only by `session_id`, making query history susceptible to leakage. **Remediation**: Add `user_id` column.
7. **`otp_sessions`** (`sol.db`): Password resets/verifications. Currently scoped by `email` string rather than foreign key. **Remediation**: Add `user_id` for strict verification.

---

## 2. The Relational Standard

### Foreign Key Constraints
For all hazard tables being scoped, we will inject the following relational standard:
```python
user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
```
> [!IMPORTANT]
> The `nullable=False` setting guarantees that orphaned records cannot exist. The `ondelete="CASCADE"` constraint ensures that if a user deletes their account, all related data (settings, tasks, runs, tokens) are automatically and completely expunged from the database, satisfying GDPR/privacy requirements.

### User Onboarding Lifecycle
To ensure default configurations (`Settings` and `Dashboard`) are provisioned instantly upon user registration:
1. **Database Event Listeners (SQLAlchemy):** We will use the `@event.listens_for(User, 'after_insert')` hook.
2. **Atomic Transaction:** The listener automatically executes in the same database session, inserting:
   - A row in `user_settings` with empty/default API keys.
   - A row in `dashboard_preferences` with default layout configurations.
3. **Rollback Safety:** If the user registration fails or rolls back, the provisioned settings are automatically rolled back, preventing dangling config rows.

---

## 3. Data Migration Strategy (The Bridge)

To migrate live databases without violating `NOT NULL` or foreign key constraints:

1. **Default Owner Assignment:**
   Identify or seed a Default/System User (e.g., User ID `1` or the primary administrator).
2. **Incremental Migration Script:**
   - Add new columns as `nullable=True` first.
   - Run an update script to populate the `user_id` field:
     - For `responses`, join `forms` to inherit the owner's `user_id`.
     - For `task_runs`, join `tasks` and `projects` to inherit the `user_id`.
     - For `token_usage_records` and `search_logs`, default them to User ID `1` or system default.
     - For `user_settings`, extract the current global `.env` values, seed them for User ID `1`, and then wipe them from `.env` to secure the system.
   - Alter the columns to `nullable=False`.

---

## 4. Step-by-Step Execution Roadmap

### Phase 1: ORM Schema updates
* Define the `UserSettings` and `DashboardPreferences` models in `backend/models.py`.
* Add `user_id` columns to `OTPSession`, `FormResponse`, `TaskRun`, `TokenUsageRecord`, `SearchLog`, and `Recommendation`.
* Register SQLAlchemy event listeners for the `User` lifecycle to auto-provision records.

### Phase 2: DB Migration generation
* Create a python migration script (`backend/migrate_multitenancy.py`) that:
  - Alters tables to add columns.
  - Migrates existing records (mapping relationships to inherit owner IDs).
  - Enforces `NOT NULL` constraints.
  - Seeds default settings rows for all existing users.

### Phase 3: Repository/Service Layer Query Refactoring
* **Settings Endpoints**: Refactor `/api/settings/credentials` and `/api/settings/check-credentials` to read and write from `UserSettings` model filtered by `current_user.id`.
* **Dashboard Endpoints**: Refactor `/api/dashboard/stats` to read layout and metrics scoped exclusively to `current_user.id`.
* **Query Verification**: Conduct code review of all route files to ensure no un-scoped `db.query(...)` calls exist on hazard tables.
