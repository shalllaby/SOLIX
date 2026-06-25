import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import SessionLocal
from backend.models import User
from backend.auth import get_password_hash

def main():
    db = SessionLocal()
    try:
        email = "antigravity@example.com"
        password = "Password123!"
        hashed_pwd = get_password_hash(password)
        
        # Check if already exists
        user = db.query(User).filter(User.email == email).first()
        if user:
            print(f"User {email} already exists. Resetting password and activating...")
            user.hashed_password = hashed_pwd
            user.status = "active"
            db.commit()
            print("User updated and activated successfully.")
            return
            
        new_user = User(
            first_name="Anti",
            last_name="Gravity",
            username="antigravity",
            email=email,
            hashed_password=hashed_pwd,
            status="active"
        )
        db.add(new_user)
        db.commit()
        print(f"Created and activated new user: {email} with password: {password}")
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
