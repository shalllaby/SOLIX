import sys
import os

# Add parent directory to path to allow importing from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal, Base, engine
from backend.models import User, JobRecord

client = TestClient(app)

def test_dashboard_flow():
    print("=== Testing Dashboard Integration Flow ===")
    
    # 1. Register a test user
    test_email = f"test_dashboard_{os.urandom(4).hex()}@example.com"
    register_payload = {
        "email": test_email,
        "password": "Password123!",
        "first_name": "Test",
        "last_name": "User"
    }
    
    print(f"Registering user: {test_email}")
    reg_response = client.post("/api/v1/auth/register", json=register_payload)
    if reg_response.status_code != 200:
        print(f"Registration failed: {reg_response.status_code} - {reg_response.text}")
        return False
    
    # Activate user to bypass OTP verification
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == test_email).first()
        if not user:
            print("Registered user not found in DB!")
            return False
        user.status = "active"
        db.commit()
        print("User activated in database.")
    except Exception as e:
        print(f"Error activating user: {e}")
        db.rollback()
        return False
    finally:
        db.close()
    
    # 2. Login to get token
    print("Logging in...")
    login_payload = {
        "email": test_email,
        "password": "Password123!"
    }
    login_response = client.post("/api/v1/auth/login", json=login_payload)
    if login_response.status_code != 200:
        print(f"Login failed: {login_response.status_code} - {login_response.text}")
        return False
    
    token_data = login_response.json()
    token = token_data.get("token")
    headers = {"Authorization": f"Bearer {token}"}
    print("Login successful! Token acquired.")

    # 3. Create some dummy job records in SQLite to check stats counts
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == test_email).first()
        if not user:
            print("Registered user not found in DB!")
            return False
            
        print("Inserting mock JobRecord entries in DB...")
        job1 = JobRecord(
            task_id="mock-task-1",
            user_id=user.id,
            task_type="cleaning",
            filename="dirty_sales.csv",
            file_size_bytes=1048576, # 1MB
            row_count=1000,
            col_count=10,
            strategy="Beta",
            status="completed",
            accuracy_rate=98.5
        )
        job2 = JobRecord(
            task_id="mock-task-2",
            user_id=user.id,
            task_type="synthetic",
            filename="synthetic_customers.csv",
            file_size_bytes=2097152, # 2MB
            row_count=5000,
            col_count=15,
            strategy="Tvae",
            status="completed",
            accuracy_rate=92.0
        )
        job3 = JobRecord(
            task_id="mock-task-3",
            user_id=user.id,
            task_type="cleaning",
            filename="active_stream.csv",
            file_size_bytes=512000, # 500KB
            row_count=2000,
            col_count=8,
            strategy="Alpha",
            status="processing"
        )
        db.add_all([job1, job2, job3])
        db.commit()
        print("Mock jobs successfully written to database.")
    except Exception as e:
        print(f"Error inserting mock jobs: {e}")
        db.rollback()
        return False
    finally:
        db.close()

    # 4. Fetch dashboard stats
    print("Querying /api/dashboard/stats...")
    stats_response = client.get("/api/dashboard/stats", headers=headers)
    if stats_response.status_code != 200:
        print(f"Failed to fetch stats: {stats_response.status_code} - {stats_response.text}")
        return False
        
    res = stats_response.json()
    print("Stats Response matches format!")
    print(f"Response data: {res}")
    
    # 5. Assertions
    stats = res.get("stats", {})
    assert stats.get("total_cleaned") == 1, f"Expected 1 cleaned, got {stats.get('total_cleaned')}"
    assert stats.get("total_synthetic") == 1, f"Expected 1 synthetic, got {stats.get('total_synthetic')}"
    assert stats.get("active_jobs_count") == 1, f"Expected 1 active, got {stats.get('active_jobs_count')}"
    assert stats.get("total_size_bytes") == 3657728, f"Expected size 3657728, got {stats.get('total_size_bytes')}"
    assert abs(stats.get("avg_accuracy_rate") - 95.25) < 0.01, f"Expected avg accuracy 95.25, got {stats.get('avg_accuracy_rate')}"
    
    recent_jobs = res.get("recent_jobs", [])
    assert len(recent_jobs) == 3, f"Expected 3 recent jobs, got {len(recent_jobs)}"
    
    chart_data = res.get("chart_data", {})
    assert "timeline" in chart_data, "Timeline chart data missing"
    assert "type_distribution" in chart_data, "Type distribution chart data missing"
    
    print("\n[SUCCESS] All backend integration assertions passed perfectly!")
    return True

if __name__ == "__main__":
    success = test_dashboard_flow()
    sys.exit(0 if success else 1)
