import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, timedelta
import unittest.mock as mock
import sys
import os

# Ensure the run directory is in path
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app
from backend.database import Base, get_db
from backend.models import User, OTPSession, AuthLog
import backend.workers.email_worker as email_worker

# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency override
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # Setup: Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown: Drop tables
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def test_register_and_otp_flow():
    # Mock the email sender so we don't try sending real emails during test,
    # but we can spy on the generated OTP code.
    with mock.patch.object(email_worker, "send_otp_email", return_value=True) as mock_send_email:
        # 1. Register a new user
        reg_payload = {
            "first_name": "tester",
            "last_name": "user",
            "email": "test@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }
        response = client.post("/api/v1/auth/register", json=reg_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "verify-otp" in data["redirect_url"]
        
        # Verify user is created in database as pending_verification
        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == "test@example.com").first()
        assert user is not None
        assert user.status == "pending_verification"
        
        # Verify email background task was triggered
        mock_send_email.assert_called_once()
        called_args = mock_send_email.call_args[0]
        # send_otp_email(recipient_email, username, otp)
        assert called_args[0] == "test@example.com"
        assert called_args[1] == "tester"
        otp_code = called_args[2]
        assert len(otp_code) == 6
        
        # 2. Try to login while pending verification (should be blocked)
        login_payload = {
            "email": "test@example.com",
            "password": "Password123!"
        }
        login_resp = client.post("/api/v1/auth/login", json=login_payload)
        assert login_resp.status_code == 403
        assert "not verified" in login_resp.json()["detail"].lower()
        
        # 3. Verify OTP with correct code
        verify_payload = {
            "email": "test@example.com",
            "otp": otp_code
        }
        verify_resp = client.post("/api/v1/auth/verify-otp", json=verify_payload)
        assert verify_resp.status_code == 200
        assert verify_resp.json()["status"] == "success"
        
        # Check secure HttpOnly cookie is set in response
        assert "sol_auth_token" in verify_resp.cookies
        
        # Verify user status is now active
        db.refresh(user)
        assert user.status == "active"
        
        # 4. Try to verify OTP again (replay protection)
        replay_resp = client.post("/api/v1/auth/verify-otp", json=verify_payload)
        assert replay_resp.status_code == 400
        assert "already been verified" in replay_resp.json()["detail"]
        
        # 5. Login successfully now that account is active
        login_resp2 = client.post("/api/v1/auth/login", json=login_payload)
        assert login_resp2.status_code == 200
        assert "sol_auth_token" in login_resp2.cookies
        db.close()

def test_password_strength_validation():
    # Test weak password (too short)
    payload_short = {
        "first_name": "tester2",
        "last_name": "user",
        "email": "test2@example.com",
        "password": "P1!",
        "confirm_password": "P1!"
    }
    response = client.post("/api/v1/auth/register", json=payload_short)
    assert response.status_code == 422
    
    # Test weak password (no digit)
    payload_no_digit = {
        "first_name": "tester2",
        "last_name": "user",
        "email": "test2@example.com",
        "password": "Password!",
        "confirm_password": "Password!"
    }
    response = client.post("/api/v1/auth/register", json=payload_no_digit)
    assert response.status_code == 422
    
    # Test weak password (no special character)
    payload_no_special = {
        "first_name": "tester2",
        "last_name": "user",
        "email": "test2@example.com",
        "password": "Password123",
        "confirm_password": "Password123"
    }
    response = client.post("/api/v1/auth/register", json=payload_no_special)
    assert response.status_code == 422
    
    # Test mismatched passwords
    payload_mismatch = {
        "first_name": "tester2",
        "last_name": "user",
        "email": "test2@example.com",
        "password": "Password123!",
        "confirm_password": "Password123!!d"
    }
    response = client.post("/api/v1/auth/register", json=payload_mismatch)
    assert response.status_code == 422

def test_otp_failed_attempts_and_lockout():
    with mock.patch.object(email_worker, "send_otp_email", return_value=True):
        reg_payload = {
            "first_name": "lockout_tester",
            "last_name": "user",
            "email": "lockout@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }
        client.post("/api/v1/auth/register", json=reg_payload)
        
        # Submit invalid OTP 5 times
        verify_payload = {
            "email": "lockout@example.com",
            "otp": "999999"
        }
        
        for i in range(4):
            resp = client.post("/api/v1/auth/verify-otp", json=verify_payload)
            assert resp.status_code == 400
            assert "attempts remaining" in resp.json()["detail"]
            
        # The 5th attempt should lock out the session
        resp_5 = client.post("/api/v1/auth/verify-otp", json=verify_payload)
        assert resp_5.status_code == 403
        assert "attempts exceeded" in resp_5.json()["detail"]

        # Even with correct OTP (if we manually query it), it should remain locked out
        db = TestingSessionLocal()
        otp_sess = db.query(OTPSession).filter(OTPSession.email == "lockout@example.com").first()
        assert otp_sess.attempts >= 5
        db.close()

def test_otp_expiration():
    with mock.patch.object(email_worker, "send_otp_email", return_value=True) as mock_send_email:
        reg_payload = {
            "first_name": "expire_tester",
            "last_name": "user",
            "email": "expire@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }
        client.post("/api/v1/auth/register", json=reg_payload)
        otp_code = mock_send_email.call_args[0][2]
        
        # Artificially set expiration time in the past
        db = TestingSessionLocal()
        otp_sess = db.query(OTPSession).filter(OTPSession.email == "expire@example.com").first()
        otp_sess.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
        db.close()
        
        # Verify expiration triggers error
        verify_payload = {
            "email": "expire@example.com",
            "otp": otp_code
        }
        resp = client.post("/api/v1/auth/verify-otp", json=verify_payload)
        assert resp.status_code == 400
        assert "expired" in resp.json()["detail"]

def test_resend_cooldown():
    with mock.patch.object(email_worker, "send_otp_email", return_value=True):
        reg_payload = {
            "first_name": "resend_tester",
            "last_name": "user",
            "email": "resend@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }
        client.post("/api/v1/auth/register", json=reg_payload)
        
        # First resend immediate attempt should trigger cooldown error (cooldown is 60s)
        resend_resp = client.post("/api/v1/auth/resend-otp", json={"email": "resend@example.com"})
        assert resend_resp.status_code == 429
        assert "seconds" in resend_resp.json()["detail"]

def test_rate_limiting():
    # Make multiple quick requests to verify the rate limiting middleware triggers 429
    # The middleware limit is set to 30 requests per minute per IP.
    # Let's verify by making 31 requests to /api/v1/auth/login.
    login_payload = {
        "email": "ratelimit@example.com",
        "password": "Password123!"
    }
    
    triggered = False
    for _ in range(40):
        resp = client.post("/api/v1/auth/login", json=login_payload, headers={"x-test-ip": "ratelimit-ip"})
        if resp.status_code == 429:
            triggered = True
            break
            
    assert triggered is True

def test_email_dispatch_called():
    with mock.patch.object(email_worker, "send_otp_email", return_value=True) as mock_send_email:
        reg_payload = {
            "first_name": "dispatch_tester",
            "last_name": "user",
            "email": "dispatch@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }
        response = client.post("/api/v1/auth/register", json=reg_payload)
        assert response.status_code == 200
        
        # Validate call arguments
        mock_send_email.assert_called_once()
        called_args = mock_send_email.call_args[0]
        # send_otp_email(recipient_email, username, otp)
        assert called_args[0] == "dispatch@example.com"
        otp_code = called_args[2]
        assert len(otp_code) == 6

# --- OAuth Tests ---

def test_google_login_redirect():
    response = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert response.status_code in [302, 307]
    location = response.headers.get("location")
    assert "accounts.google.com" in location
    assert "client_id=" in location
    assert "redirect_uri=" in location

def test_github_login_redirect():
    response = client.get("/api/v1/auth/github/login", follow_redirects=False)
    assert response.status_code in [302, 307]
    location = response.headers.get("location")
    assert "github.com" in location
    assert "client_id=" in location
    assert "redirect_uri=" in location

def test_google_callback_flow():
    mock_token_resp = mock.Mock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = {"access_token": "mock-google-token"}
    
    mock_userinfo_resp = mock.Mock()
    mock_userinfo_resp.status_code = 200
    mock_userinfo_resp.json.return_value = {
        "email": "oauth_google@example.com",
        "name": "Google OAuth User"
    }
    
    async def mock_post(*args, **kwargs):
        return mock_token_resp
        
    async def mock_get(*args, **kwargs):
        return mock_userinfo_resp
        
    with mock.patch("httpx.AsyncClient.post", side_effect=mock_post), \
         mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
         
        response = client.get("/api/v1/auth/google/callback?code=mock_code", follow_redirects=False)
        assert response.status_code in [302, 307]
        assert response.headers.get("location") == "/app/dashboard"
        assert "sol_auth_token" in response.cookies
        
        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == "oauth_google@example.com").first()
        assert user is not None
        assert user.status == "active"
        assert user.username == "google_oauth_user" or user.username.startswith("google_oauth")
        db.close()

def test_github_callback_flow():
    mock_token_resp = mock.Mock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = {"access_token": "mock-github-token"}
    
    mock_user_resp = mock.Mock()
    mock_user_resp.status_code = 200
    mock_user_resp.json.return_value = {
        "login": "github_oauth_user",
        "email": "oauth_github@example.com"
    }
    
    async def mock_post(*args, **kwargs):
        return mock_token_resp
        
    async def mock_get(*args, **kwargs):
        return mock_user_resp
        
    with mock.patch("httpx.AsyncClient.post", side_effect=mock_post), \
         mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
         
        response = client.get("/api/v1/auth/github/callback?code=mock_code", follow_redirects=False)
        assert response.status_code in [302, 307]
        assert response.headers.get("location") == "/app/dashboard"
        assert "sol_auth_token" in response.cookies
        
        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == "oauth_github@example.com").first()
        assert user is not None
        assert user.status == "active"
        assert user.username == "github_oauth_user"
        db.close()
