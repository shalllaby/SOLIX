import sys
import os

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from backend.database import SessionLocal
from backend.models import User
from backend.auth import get_password_hash

def seed_user():
    db = SessionLocal()
    try:
        email = "test@example.com"
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print("Creating test user...")
            hashed_pwd = get_password_hash("Password123!")
            user = User(
                first_name="Test",
                last_name="User",
                username="testuser",
                email=email,
                hashed_password=hashed_pwd,
                status="active"
            )
            db.add(user)
            db.commit()
            print("Test user created successfully!")
        else:
            print("Test user already exists.")
            if user.status != "active":
                print("Activating existing test user...")
                user.status = "active"
                db.commit()
                print("Activated.")
    except Exception as e:
        print(f"Error seeding user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_user()
