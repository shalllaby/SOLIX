import os
import sys
import sqlite3
import uuid
from datetime import datetime

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, Base, SessionLocal
from backend.models import User, ChatSession, ChatMessage

def run_migration():
    print("Starting Chatbot Multi-Session Database Migration...")
    
    # 1. Create tables if they do not exist
    Base.metadata.create_all(bind=engine)
    print("Created chat_sessions and chat_messages tables in sol.db (if they didn't exist).")

    db = SessionLocal()
    
    # 2. Check if legacy chatbot cache database exists
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    cache_db_path = os.path.join(backend_dir, "data", "chatbot_cache.db")
    
    if not os.path.exists(cache_db_path):
        print(f"No legacy chatbot cache database found at {cache_db_path}. Nothing to migrate.")
        db.close()
        return

    print(f"Found legacy chatbot cache database at {cache_db_path}. Starting data migration...")
    
    try:
        conn = sqlite3.connect(cache_db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_cache'")
        if not cursor.fetchone():
            print("Table 'chat_cache' not found in cache database. Nothing to migrate.")
            conn.close()
            db.close()
            return
            
        cursor.execute("SELECT query, response, created_at FROM chat_cache")
        rows = cursor.fetchall()
        print(f"Retrieved {len(rows)} legacy cache records.")
        
        # Map user_id to session UUID
        user_sessions = {}
        migrated_count = 0
        
        for query_key, response, created_at_str in rows:
            # Parse user_id and actual query from query_key. Format: user_<user_id>:<query>
            if not query_key.startswith("user_"):
                continue
                
            try:
                parts = query_key.split(":", 1)
                user_part = parts[0]
                actual_query = parts[1] if len(parts) > 1 else ""
                user_id = int(user_part.split("_")[1])
            except (ValueError, IndexError) as e:
                print(f"Skipping unparseable query key: '{query_key}' ({e})")
                continue
                
            # Verify if user exists in sol.db
            user_exists = db.query(User).filter(User.id == user_id).first()
            if not user_exists:
                print(f"Skipping legacy records for user_id={user_id} (user no longer exists in DB).")
                continue
                
            # Parse date
            created_at = datetime.utcnow()
            if created_at_str:
                try:
                    # sqlite TIMESTAMP can be 'YYYY-MM-DD HH:MM:SS.ffffff' or similar
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Retrieve or create an initial session for this user
            if user_id not in user_sessions:
                session_id = str(uuid.uuid4())
                new_session = ChatSession(
                    id=session_id,
                    user_id=user_id,
                    title="Archived Session (Migration)",
                    created_at=created_at,
                    updated_at=created_at
                )
                db.add(new_session)
                user_sessions[user_id] = session_id
                print(f"Created initial chat session '{session_id}' for user_id={user_id}")
            
            session_id = user_sessions[user_id]
            
            # Insert User Message
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=actual_query,
                created_at=created_at
            )
            db.add(user_msg)
            
            # Insert Bot Message
            bot_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="bot",
                content=response,
                created_at=created_at
            )
            db.add(bot_msg)
            
            migrated_count += 1
            
        db.commit()
        print(f"Successfully migrated {migrated_count} legacy conversation pairs into modern multi-session records.")
        conn.close()
        
    except Exception as e:
        db.rollback()
        print(f"Migration failed: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    run_migration()
