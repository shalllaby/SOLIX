import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
import sys
import os

# Ensure the run directory is in path
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app
from backend.database import Base, get_db
from backend.models import User, Project, Task, Form, FormResponse, JobRecord, UserSettings
from backend.auth import create_access_token

# Use in-memory SQLite for isolated testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # Setup
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def create_test_user(db_session, email, password, is_admin=False):
    # Hash password (just using plain text for testing speed or mock it)
    user = User(
        first_name="Test",
        last_name="User",
        email=email,
        hashed_password="mocked_password",
        status="active",
        is_admin=is_admin
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token(data={"sub": email})
    return user, token

def test_project_and_task_isolation():
    db = TestingSessionLocal()
    
    # 1. Create two users
    user_a, token_a = create_test_user(db, "usera@example.com", "password")
    user_b, token_b = create_test_user(db, "userb@example.com", "password")
    
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # 2. User A fetches projects (should auto-seed one project for User A)
    res_a = client.get("/api/projects", headers=headers_a)
    assert res_a.status_code == 200
    projects_a = res_a.json()
    assert len(projects_a) == 1
    proj_a_id = projects_a[0]["id"]
    
    # 3. User B fetches projects (should auto-seed one project for User B)
    res_b = client.get("/api/projects", headers=headers_b)
    assert res_b.status_code == 200
    projects_b = res_b.json()
    assert len(projects_b) == 1
    proj_b_id = projects_b[0]["id"]
    
    # Assert they are different projects
    assert proj_a_id != proj_b_id
    
    # 4. User B tries to create a task in User A's project (should return 403)
    res_task = client.post(
        "/api/tasks",
        headers=headers_b,
        json={"name": "Malicious Task", "project_id": proj_a_id, "state_data": {}}
    )
    assert res_task.status_code == 403
    
    # 5. User A creates a task in User A's project (should succeed)
    res_task_ok = client.post(
        "/api/tasks",
        headers=headers_a,
        json={"name": "Good Task", "project_id": proj_a_id, "state_data": {}}
    )
    assert res_task_ok.status_code == 200
    
    db.close()

def test_form_isolation_and_idor():
    db = TestingSessionLocal()
    
    # 1. Create users
    user_a, token_a = create_test_user(db, "usera_form@example.com", "password")
    user_b, token_b = create_test_user(db, "userb_form@example.com", "password")
    
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # 2. User A creates a form
    res_create = client.post(
        "/api/forms",
        headers=headers_a,
        json={
            "title": "Feedback Form",
            "description": "User A Feedback Form",
            "questions": [{"type": "text", "question": "Your name?", "options": []}]
        }
    )
    assert res_create.status_code == 200
    form_id = res_create.json()["form_id"]
    
    # 3. User B lists forms (should be empty for User B)
    res_list_b = client.get("/api/forms", headers=headers_b)
    assert res_list_b.status_code == 200
    assert len(res_list_b.json()["forms"]) == 0
    
    # 4. User B tries to access User A's form responses (should return 403)
    res_resp_b = client.get(f"/api/forms/{form_id}/responses", headers=headers_b)
    assert res_resp_b.status_code == 403
    
    # 5. User B tries to export User A's form responses (should return 403)
    res_export_b = client.get(f"/api/forms/{form_id}/export", headers=headers_b)
    assert res_export_b.status_code == 403
    
    # 6. User B tries to update User A's form (should return 403)
    res_update_b = client.put(
        f"/api/forms/{form_id}",
        headers=headers_b,
        json={
            "title": "Hacked Form",
            "description": "Hacked",
            "questions": []
        }
    )
    assert res_update_b.status_code == 403
    
    # 7. User B tries to delete User A's form (should return 403)
    res_delete_b = client.delete(f"/api/forms/{form_id}", headers=headers_b)
    assert res_delete_b.status_code == 403
    
    # 8. User A gets their own form responses (should succeed)
    res_resp_a = client.get(f"/api/forms/{form_id}/responses", headers=headers_a)
    assert res_resp_a.status_code == 200
    
    db.close()

def test_admin_dashboard_authorization():
    db = TestingSessionLocal()
    
    # 1. Create standard and admin users
    standard_user, token_standard = create_test_user(db, "normal@example.com", "password", is_admin=False)
    admin_user, token_admin = create_test_user(db, "admin_test@example.com", "password", is_admin=True)
    
    headers_std = {"Authorization": f"Bearer {token_standard}"}
    headers_adm = {"Authorization": f"Bearer {token_admin}"}
    
    # 2. Standard user tries to hit admin stats (should return 403)
    res_stats_std = client.get("/api/admin/stats", headers=headers_std)
    assert res_stats_std.status_code == 403
    
    # 3. Admin user hits admin stats (should succeed)
    res_stats_adm = client.get("/api/admin/stats", headers=headers_adm)
    assert res_stats_adm.status_code == 200
    
    # 4. Standard user tries to fetch users list (should return 403)
    res_users_std = client.get("/api/admin/users", headers=headers_std)
    assert res_users_std.status_code == 403
    
    # 5. Admin user fetches users list (should succeed)
    res_users_adm = client.get("/api/admin/users", headers=headers_adm)
    assert res_users_adm.status_code == 200
    assert len(res_users_adm.json()) >= 2 # Admin + standard user
    
    db.close()

def test_dashboard_stats_isolation():
    db = TestingSessionLocal()
    
    # 1. Create standard User A and User B
    user_a, token_a = create_test_user(db, "usera@example.com", "password", is_admin=False)
    user_b, token_b = create_test_user(db, "userb@example.com", "password", is_admin=False)
    
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # 2. Add a completed cleaning job for User A
    job_a = JobRecord(
        task_id="job_a_123",
        user_id=user_a.id,
        task_type="cleaning",
        status="completed",
        filename="test_dataset.csv",
        file_size_bytes=1024,
        accuracy_rate=95.0
    )
    db.add(job_a)
    db.commit()
    
    # 3. User B retrieves their dashboard stats (should be all 0/empty)
    res_b = client.get("/api/dashboard/stats", headers=headers_b)
    assert res_b.status_code == 200
    data_b = res_b.json()
    
    assert data_b["stats"]["total_cleaned"] == 0
    assert data_b["stats"]["total_synthetic"] == 0
    assert data_b["stats"]["total_automl"] == 0
    assert data_b["stats"]["total_size_bytes"] == 0
    assert data_b["stats"]["active_jobs_count"] == 0
    assert len(data_b["recent_jobs"]) == 0
    
    # 4. User A retrieves their dashboard stats (should contain their job count/details)
    res_a = client.get("/api/dashboard/stats", headers=headers_a)
    assert res_a.status_code == 200
    data_a = res_a.json()
    
    assert data_a["stats"]["total_cleaned"] == 1
    assert data_a["stats"]["total_size_bytes"] == 1024
    assert len(data_a["recent_jobs"]) == 1
    assert data_a["recent_jobs"][0]["task_id"] == "job_a_123"
    
    db.close()

def test_chatbot_cache_isolation():
    db = TestingSessionLocal()
    
    # 1. Create standard User A and User B
    user_a, token_a = create_test_user(db, "chat_usera@example.com", "password", is_admin=False)
    user_b, token_b = create_test_user(db, "chat_userb@example.com", "password", is_admin=False)
    
    # Set valid keys to bypass credentials barrier
    settings_a = db.query(UserSettings).filter_by(user_id=user_a.id).first()
    if settings_a:
        settings_a.groq_api_key = "gsk_valid_a"
    settings_b = db.query(UserSettings).filter_by(user_id=user_b.id).first()
    if settings_b:
        settings_b.groq_api_key = "gsk_valid_b"
    db.commit()
    
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    from backend.tools.chatbot.router import cache
    import sqlite3
    
    # Clean up the cache database and self.cached_queries list for our test users to ensure zero contamination
    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_cache WHERE query LIKE ? OR query LIKE ?", (f"user_{user_a.id}:%", f"user_{user_b.id}:%"))
    conn.commit()
    conn.close()
    cache.load_cache_keys()
    
    # Pre-populate some exact cached query for User A
    cache.set("hello", "User A hello response", user_id=user_a.id)
    
    # 2. User B queries "hello" (should not hit User A's cache and instead call LLM/mock or fail to find)
    assert cache.get("hello", user_id=user_a.id) == "User A hello response"
    assert cache.get("hello", user_id=user_b.id) is None

    # Let's test the endpoint as well using client
    from unittest.mock import patch
    with patch("requests.post") as mock_post:
        # Mocking external API response for chatbot
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "User B custom response"}}]
        }
        
        # User B posts "hello"
        res_b = client.post("/api/chat", json={"message": "hello"}, headers=headers_b)
        assert res_b.status_code == 200
        assert res_b.json()["response"] == "User B custom response"
        assert res_b.json()["cached"] is False
        
        # Now check that User B's exact query was cached for B
        assert cache.get("hello", user_id=user_b.id) == "User B custom response"
        
        # And check User A still gets their original cached response (cached=True)
        res_a = client.post("/api/chat", json={"message": "hello"}, headers=headers_a)
        assert res_a.status_code == 200
        assert res_a.json()["response"] == "User A hello response"
        assert res_a.json()["cached"] is True

    db.close()

def test_in_memory_store_isolation():
    db = TestingSessionLocal()
    
    user_a, token_a = create_test_user(db, "store_a@example.com", "password", is_admin=False)
    user_b, token_b = create_test_user(db, "store_b@example.com", "password", is_admin=False)
    
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # 1. Simulate User A uploading a dataset
    res_upload_a = client.post(
        "/api/synthetic/upload",
        files={"file": ("test.csv", "col1,col2\n1,2\n3,4")},
        headers=headers_a
    )
    assert res_upload_a.status_code == 200
    dataset_id_a = res_upload_a.json()["dataset_id"]
    
    # 2. Simulate User B uploading a dataset
    res_upload_b = client.post(
        "/api/synthetic/upload",
        files={"file": ("other.csv", "colA,colB\n5,6\n7,8")},
        headers=headers_b
    )
    assert res_upload_b.status_code == 200
    dataset_id_b = res_upload_b.json()["dataset_id"]
    
    # 3. List datasets for User B (should only contain other.csv, not test.csv!)
    res_list_b = client.get("/api/synthetic/datasets", headers=headers_b)
    assert res_list_b.status_code == 200
    datasets_b = res_list_b.json()
    assert len(datasets_b) == 1
    assert datasets_b[0]["filename"] == "other.csv"
    assert datasets_b[0]["dataset_id"] == dataset_id_b
    
    # 4. List datasets for User A (should only contain test.csv!)
    res_list_a = client.get("/api/synthetic/datasets", headers=headers_a)
    assert res_list_a.status_code == 200
    datasets_a = res_list_a.json()
    assert len(datasets_a) == 1
    assert datasets_a[0]["filename"] == "test.csv"
    assert datasets_a[0]["dataset_id"] == dataset_id_a
    
    # 5. User B tries to download User A's dataset (should return 404/not found!)
    res_download_b = client.get(f"/api/synthetic/download/{dataset_id_a}", headers=headers_b)
    assert res_download_b.status_code == 404
    
    # 6. User A downloads User A's dataset (should succeed)
    res_download_a = client.get(f"/api/synthetic/download/{dataset_id_a}", headers=headers_a)
    assert res_download_a.status_code == 200

    db.close()

def test_credentials_barrier_enforcement():
    db = TestingSessionLocal()
    
    # 1. Create User A (settings record is auto-seeded by SQLAlchemy listener)
    user_a, token_a = create_test_user(db, "barrier_usera@example.com", "password")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    
    # Clear the auto-seeded key so User A has no configured key
    settings_a = db.query(UserSettings).filter_by(user_id=user_a.id).first()
    assert settings_a is not None
    settings_a.groq_api_key = ""
    db.commit()
    
    # User A tries to call chatbot /api/chat - should fail with 403 Forbidden
    res_a = client.post("/api/chat", json={"message": "hello"}, headers=headers_a)
    assert res_a.status_code == 403
    assert "Remote Credentials Required" in res_a.json()["detail"]
    
    # 2. Add User A settings with empty key
    res_a_empty = client.post("/api/chat", json={"message": "hello"}, headers=headers_a)
    assert res_a_empty.status_code == 403
    
    # 3. Add User A settings with placeholder key
    settings_a.groq_api_key = "your_api_key_here"
    db.commit()
    
    res_a_placeholder = client.post("/api/chat", json={"message": "hello"}, headers=headers_a)
    assert res_a_placeholder.status_code == 403
    
    # 4. Set valid key for User A
    settings_a.groq_api_key = "gsk_validkey"
    db.commit()
    
    # Mock requests.post to avoid hitting actual Groq API
    from unittest.mock import patch
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Mocked LLM reply"}}]
        }
        res_a_valid = client.post("/api/chat", json={"message": "hello"}, headers=headers_a)
        assert res_a_valid.status_code == 200
        
    db.close()
