import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Path wiring
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from backend.database import SessionLocal, engine
from backend.models import User
from backend.auth import get_password_hash

db = SessionLocal()
try:
    users = db.query(User).all()
    print("Found users:")
    for u in users:
        print(f"- ID: {u.id}, Email: {u.email}, Username: {u.username}, Status: {u.status}")
        
    # Check if test@example.com exists, if not create it
    test_user = db.query(User).filter(User.email == "test@example.com").first()
    if not test_user:
        print("Creating test@example.com active user...")
        hashed_pwd = get_password_hash("Test12345!")
        new_user = User(
            first_name="Test",
            last_name="User",
            username="testuser",
            email="test@example.com",
            hashed_password=hashed_pwd,
            status="active"
        )
        db.add(new_user)
        db.commit()
        print("User created successfully!")
    else:
        # Make sure it's active
        if test_user.status != "active":
            print("Activating test@example.com...")
            test_user.status = "active"
            db.commit()
            print("Activated.")
finally:
    db.close()
