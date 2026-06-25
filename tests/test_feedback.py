import os
import sys
import pytest
from fastapi.testclient import TestClient

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app
from backend.models import Feedback
from backend.database import SessionLocal

client = TestClient(app)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()

def test_get_feedback_page():
    """Verify that the feedback HTML page loads successfully."""
    response = client.get("/feedback")
    assert response.status_code == 200
    assert "Share Your Experience" in response.text
    assert "feedback-form" in response.text

def test_submit_feedback_success(db_session):
    """Verify that valid feedback data is saved successfully to the database."""
    # Count before
    count_before = db_session.query(Feedback).count()

    payload = {
        "name": "Test User",
        "email": "test@example.com",
        "phone": "+1234567890",
        "message": "This is a test feedback message with enough length."
    }
    response = client.post("/api/feedback", data=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Feedback submitted successfully"}

    # Count after
    count_after = db_session.query(Feedback).count()
    assert count_after == count_before + 1

    # Verify content in DB
    created = db_session.query(Feedback).filter(Feedback.email == "test@example.com").first()
    assert created is not None
    assert created.name == "Test User"
    assert created.message == "This is a test feedback message with enough length."

    # Cleanup test feedback
    db_session.delete(created)
    db_session.commit()

def test_submit_feedback_validation_errors():
    """Verify that validation errors are returned for bad input data."""
    # Missing name/email/message
    payload = {
        "name": "",
        "email": "bademail",
        "phone": "",
        "message": ""
    }
    response = client.post("/api/feedback", data=payload)
    assert response.status_code == 400

    # Bad email format
    payload = {
        "name": "Test User",
        "email": "invalid-email-format",
        "phone": "",
        "message": "Some test feedback long enough."
    }
    response = client.post("/api/feedback", data=payload)
    assert response.status_code == 400

def test_admin_portal_protection():
    """Verify that the admin portal is protected without auth cookie."""
    response = client.get("/feedback-admin-portal-solix")
    assert response.status_code == 200
    # Should render the login passcode prompt, not the feedback list
    assert "Admin Authentication" in response.text
    assert "Feedback Hub" not in response.text

def test_admin_portal_login_failure():
    """Verify that incorrect passcode redirects with error parameter."""
    response = client.post("/feedback-admin-portal-solix/login", data={"password": "wrongpassword"}, allow_redirects=False)
    assert response.status_code == 303
    assert "/feedback-admin-portal-solix?error=1" in response.headers["location"]

def test_admin_portal_login_success_and_delete(db_session):
    """Verify that correct passcode sets session cookie and unlocks delete capability."""
    # Create a dummy feedback to delete
    dummy = Feedback(name="Dummy User", email="dummy@test.com", message="Dummy message to delete")
    db_session.add(dummy)
    db_session.commit()
    db_session.refresh(dummy)
    dummy_id = dummy.id

    # 1. Login
    login_res = client.post("/feedback-admin-portal-solix/login", data={"password": "10mohamed10"}, allow_redirects=False)
    assert login_res.status_code == 303
    assert "solix_admin_auth" in login_res.cookies

    # 2. Get Admin Portal with session cookie
    portal_res = client.get("/feedback-admin-portal-solix", cookies=login_res.cookies)
    assert portal_res.status_code == 200
    assert "Feedback Hub" in portal_res.text
    assert "Dummy User" in portal_res.text

    # 3. Delete feedback using session cookie
    delete_res = client.post(f"/feedback-admin-portal-solix/delete/{dummy_id}", cookies=login_res.cookies)
    assert delete_res.status_code == 200
    assert delete_res.json()["status"] == "success"

    # Verify deleted from DB
    deleted = db_session.query(Feedback).filter(Feedback.id == dummy_id).first()
    assert deleted is None
