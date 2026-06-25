import sqlite3
import os
import traceback

def run_diagnostic():
    log_path = os.path.join(os.path.dirname(__file__), "..", "migration_log.txt")
    db_path = os.path.join(os.path.dirname(__file__), "data", "sol.db")
    
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("--- MIGRATION & DATABASE DIAGNOSTIC ---\n")
        try:
            log.write(f"Connecting to database at {db_path}...\n")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            log.write(f"Existing tables: {tables}\n\n")

            # Check each table schema
            for table in tables:
                log.write(f"Table: {table}\n")
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                for col in columns:
                    log.write(f"  Column: {col}\n")
                log.write("\n")

            conn.close()
        except Exception as e:
            log.write(f"Error during diagnostic: {e}\n")
            log.write(traceback.format_exc())

if __name__ == "__main__":
    run_diagnostic()
