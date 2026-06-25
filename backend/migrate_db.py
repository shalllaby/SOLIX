import sqlite3
import os

def migrate():
    db_path = os.path.join(os.path.dirname(__file__), "data", "sol.db")
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Migrate "users" table to add new columns if missing
    cursor.execute("PRAGMA table_info(users)")
    user_columns = [row[1] for row in cursor.fetchall()]
    print("Existing users columns:", user_columns)
    if "username" not in user_columns:
        print("Adding username column to users...")
        cursor.execute("ALTER TABLE users ADD COLUMN username VARCHAR")
    if "status" not in user_columns:
        print("Adding status column to users...")
        cursor.execute("ALTER TABLE users ADD COLUMN status VARCHAR DEFAULT 'active'")
    if "job_title" not in user_columns:
        print("Adding job_title column to users...")
        cursor.execute("ALTER TABLE users ADD COLUMN job_title VARCHAR")
    if "organization" not in user_columns:
        print("Adding organization column to users...")
        cursor.execute("ALTER TABLE users ADD COLUMN organization VARCHAR")
    if "avatar_url" not in user_columns:
        print("Adding avatar_url column to users...")
        cursor.execute("ALTER TABLE users ADD COLUMN avatar_url VARCHAR")
    if "is_admin" not in user_columns:
        print("Adding is_admin column to users...")
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")

    # Update admin status for known admin emails
    cursor.execute("UPDATE users SET is_admin = 1 WHERE email IN ('solixagentic@gmail.com', 'socilaw715@luxudata.com', 'admin@example.com', 'test_admin@example.com', 'heizul@itmo.edu.pl')")

    # 2. Migrate "projects" table to add user_id if missing
    cursor.execute("PRAGMA table_info(projects)")
    project_columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in project_columns:
        print("Adding user_id column to projects...")
        cursor.execute("ALTER TABLE projects ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")

    # 3. Migrate "forms" table to add user_id if missing
    cursor.execute("PRAGMA table_info(forms)")
    form_columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in form_columns:
        print("Adding user_id column to forms...")
        cursor.execute("ALTER TABLE forms ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")

    # 4. Migrate "auth_logs" table to add new columns if missing
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_logs'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(auth_logs)")
        log_columns = [row[1] for row in cursor.fetchall()]
        print("Existing auth_logs columns:", log_columns)
        
        expected_log_cols = {
            "ip_address": "VARCHAR",
            "email": "VARCHAR",
            "event_type": "VARCHAR",
            "details": "VARCHAR",
            "timestamp": "DATETIME"
        }
        for col, col_type in expected_log_cols.items():
            if col not in log_columns:
                print(f"Adding {col} column to auth_logs...")
                cursor.execute(f"ALTER TABLE auth_logs ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()
    print("Database schema migration precheck completed successfully.")

if __name__ == "__main__":
    migrate()
