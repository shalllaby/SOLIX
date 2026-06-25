import sqlite3
import os
import sys
from dotenv import load_dotenv

# Ensure the backend directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env variables from root .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from backend.utils.security import encrypt_value

def migrate_databases():
    db_dir = os.path.join(os.path.dirname(__file__), "data")
    sol_db_path = os.path.join(db_dir, "sol.db")
    advisor_db_path = os.path.join(db_dir, "advisor.db")

    print(f"Starting Multi-Tenancy database migration...")
    print(f"Main DB: {sol_db_path}")
    print(f"Advisor DB: {advisor_db_path}")

    # ==========================================
    # 1. Main DB (sol.db) Migrations
    # ==========================================
    if os.path.exists(sol_db_path):
        conn = sqlite3.connect(sol_db_path)
        cursor = conn.cursor()

        # Create user_settings table if it doesn't exist
        print("Ensuring 'user_settings' table exists...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                kaggle_username VARCHAR,
                kaggle_key VARCHAR,
                groq_api_key VARCHAR,
                elevenlabs_api_key VARCHAR,
                elevenlabs_id VARCHAR,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Create dashboard_preferences table if it doesn't exist
        print("Ensuring 'dashboard_preferences' table exists...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                layout JSON,
                theme_preference VARCHAR DEFAULT 'dark',
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Helper to check and add columns
        def ensure_column(table_name, col_name, col_def):
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            if col_name not in columns:
                print(f"Adding column '{col_name}' to table '{table_name}'...")
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}")

        # Add user_id column to hazard tables in sol.db
        ensure_column("otp_sessions", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")
        ensure_column("responses", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")
        ensure_column("token_usage_records", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")
        ensure_column("task_runs", "user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE")

        # Migrate data: get all users and check settings/preferences
        cursor.execute("SELECT id, email, is_admin FROM users")
        users = cursor.fetchall()
        print(f"Seeding settings and preferences for {len(users)} users...")

        # Extract current global credentials from env
        env_kaggle_user = os.getenv("KAGGLE_USERNAME", "").strip()
        env_kaggle_key = os.getenv("KAGGLE_KEY", "").strip()
        env_groq = os.getenv("GROQ_API_KEY", "").strip()
        env_elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        env_elevenlabs_id = os.getenv("ELEVENLABS_ID", "").strip()

        # Encrypt settings values
        enc_kaggle_user = encrypt_value(env_kaggle_user) if env_kaggle_user else None
        enc_kaggle_key = encrypt_value(env_kaggle_key) if env_kaggle_key else None
        enc_groq = encrypt_value(env_groq) if env_groq else None
        enc_eleven_key = encrypt_value(env_elevenlabs_key) if env_elevenlabs_key else None
        enc_eleven_id = encrypt_value(env_elevenlabs_id) if env_elevenlabs_id else None

        for user_id, email, is_admin in users:
            # Check user_settings
            cursor.execute("SELECT id FROM user_settings WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                # Seed settings for the user
                # For admin/existing users, populate with current env values so they don't lose access
                cursor.execute("""
                    INSERT INTO user_settings (
                        user_id, kaggle_username, kaggle_key, groq_api_key, elevenlabs_api_key, elevenlabs_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, enc_kaggle_user, enc_kaggle_key, enc_groq, enc_eleven_key, enc_eleven_id))

            # Check dashboard_preferences
            cursor.execute("SELECT id FROM dashboard_preferences WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO dashboard_preferences (
                        user_id, layout, theme_preference, created_at, updated_at
                    ) VALUES (?, ?, ?, datetime('now'), datetime('now'))
                """, (user_id, "{}", "dark"))

        conn.commit()
        conn.close()
        print("Main DB migrations completed successfully.")
    else:
        print("Warning: Main DB file not found, skipping main migrations.")

    # ==========================================
    # 2. Advisor DB (advisor.db) Migrations
    # ==========================================
    if os.path.exists(advisor_db_path):
        conn_adv = sqlite3.connect(advisor_db_path)
        cursor_adv = conn_adv.cursor()

        # Add user_id column to search_logs table
        cursor_adv.execute("PRAGMA table_info(search_logs)")
        columns = [row[1] for row in cursor_adv.fetchall()]
        if "user_id" not in columns:
            print("Adding column 'user_id' to search_logs...")
            cursor_adv.execute("ALTER TABLE search_logs ADD COLUMN user_id INTEGER")

        conn_adv.commit()
        conn_adv.close()
        print("Advisor DB migrations completed successfully.")
    else:
        print("Warning: Advisor DB file not found, skipping advisor migrations.")

    print("All migrations completed successfully!")

if __name__ == "__main__":
    migrate_databases()
