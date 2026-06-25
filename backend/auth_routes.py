import os
import re
import secrets
import urllib.parse
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, OTPSession, AuthLog
from backend.auth import (
    get_password_hash,
    verify_password,
    generate_otp,
    hash_otp,
    create_access_token
)
import backend.workers.email_worker as email_worker

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# Configuration constants
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

OTP_EXPIRE_MINUTES = 5
OTP_RESEND_COOLDOWN_SECONDS = 60
OTP_MAX_ATTEMPTS = 5
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# --- Request/Response Schemas ---

class RegisterSchema(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

class VerifyOTPSchema(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

class ResendOTPSchema(BaseModel):
    email: EmailStr

class LoginSchema(BaseModel):
    email: EmailStr
    password: str
    remember: bool = False

# --- Helper Functions ---

def log_auth_event(db: Session, ip: Optional[str], email: Optional[str], event_type: str, details: str):
    """Log authentication and security events to the audit trail database."""
    status_val = "success" if ("success" in event_type or "verified" in event_type or event_type == "register") else "failed"
    log = AuthLog(
        ip_address=ip,
        email=email,
        action=event_type,
        status=status_val,
        details=details,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(log)
    db.commit()

def get_redirect_uri(request: Request, provider: str) -> str:
    """
    Construct redirect URI dynamically. Google/GitHub require exact port (8000).
    Normalizes local URLs to use 'localhost:8000' to match registered OAuth configs.
    Also supports custom BACKEND_BASE_URL set in the environment (.env).
    """
    base_url = os.getenv("BACKEND_BASE_URL")
    if base_url:
        return f"{base_url.rstrip('/')}/api/v1/auth/{provider}/callback"
        
    url = str(request.url_for(f"{provider}_callback"))
    if "localhost" in url or "127.0.0.1" in url:
        url = re.sub(r"(localhost|127\.0\.0\.1):\d+", r"localhost:8000", url)
    return url

# --- Endpoints ---

@router.post("/register")
async def register(
    payload: RegisterSchema, 
    request: Request,
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    ip = request.client.host if request.client else "unknown"
    
    # Check if user already exists
    existing_user_email = db.query(User).filter(User.email == payload.email).first()
    if existing_user_email:
        log_auth_event(db, ip, payload.email, "register_failed", "Email already registered.")
        raise HTTPException(status_code=400, detail="Email is already registered.")

    # Auto-generate username from first_name and last_name
    base_username = re.sub(r"[^a-zA-Z0-9_]", "", (payload.first_name + "_" + payload.last_name).lower())
    if not base_username:
        base_username = payload.email.split("@")[0]
    username = base_username
    counter = 1
    while db.query(User).filter(User.username == username).first():
        username = f"{base_username}_{counter}"
        counter += 1

    # Create inactive user pending verification
    hashed_pwd = get_password_hash(payload.password)
    new_user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        username=username,
        email=payload.email,
        hashed_password=hashed_pwd,
        status="pending_verification"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Generate Secure OTP
    otp = generate_otp()
    hashed_code = hash_otp(otp)
    expire_time = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)

    # Upsert OTP Session
    otp_session = db.query(OTPSession).filter(OTPSession.email == payload.email).first()
    if otp_session:
        otp_session.otp_hash = hashed_code
        otp_session.expires_at = expire_time
        otp_session.attempts = 0
        otp_session.is_verified = False
        otp_session.created_at = datetime.now(timezone.utc)
    else:
        otp_session = OTPSession(
            email=payload.email,
            otp_hash=hashed_code,
            expires_at=expire_time,
            attempts=0,
            is_verified=False
        )
        db.add(otp_session)
    db.commit()

    log_auth_event(db, ip, payload.email, "register", "User account registered (pending verification).")
    log_auth_event(db, ip, payload.email, "otp_sent", "Verification OTP generated and queued.")

    # Send OTP in background
    background_tasks.add_task(email_worker.send_otp_email, payload.email, payload.first_name, otp)

    # Developer security debug logging
    print(f"\n[SECURITY DEBUG] Generated OTP for {payload.email}: {otp}\n", flush=True)
    try:
        with open("otp_debug.txt", "w") as f:
            f.write(f"Email: {payload.email}\nOTP: {otp}\nTimestamp: {datetime.now(timezone.utc)}\n")
    except Exception as e:
        print(f"[SECURITY DEBUG] Could not write to otp_debug.txt: {e}", flush=True)

    return {
        "status": "success",
        "message": "Registration successful. Please check your email for the verification code.",
        "email": payload.email,
        "redirect_url": f"/verify-otp?email={payload.email}"
    }

@router.post("/verify-otp")
async def verify_otp(
    payload: VerifyOTPSchema, 
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    ip = request.client.host if request.client else "unknown"
    
    otp_session = db.query(OTPSession).filter(OTPSession.email == payload.email).first()
    if not otp_session:
        log_auth_event(db, ip, payload.email, "otp_failed", "No verification session found.")
        raise HTTPException(status_code=400, detail="Verification session not found. Please register first.")

    # Lockout Check
    if otp_session.attempts >= OTP_MAX_ATTEMPTS:
        log_auth_event(db, ip, payload.email, "lockout", "Verification locked out due to too many failed attempts.")
        raise HTTPException(
            status_code=403, 
            detail="Maximum verification attempts exceeded. Please request a new OTP code."
        )

    # Expiration Check
    expiration_time = otp_session.expires_at
    if expiration_time.tzinfo is None:
        expiration_time = expiration_time.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expiration_time:
        log_auth_event(db, ip, payload.email, "otp_failed", "Expired OTP submitted.")
        raise HTTPException(status_code=400, detail="The verification code has expired. Please request a new OTP.")

    # Replay protection
    if otp_session.is_verified:
        log_auth_event(db, ip, payload.email, "otp_failed", "Replay attack detected: OTP already verified.")
        raise HTTPException(status_code=400, detail="This code has already been verified. Replay attack prevented.")

    # Verify code
    submitted_hashed_otp = hash_otp(payload.otp)
    if otp_session.otp_hash != submitted_hashed_otp:
        otp_session.attempts += 1
        db.commit()
        
        remaining = OTP_MAX_ATTEMPTS - otp_session.attempts
        log_auth_event(db, ip, payload.email, "otp_failed", f"Incorrect OTP. Attempts remaining: {remaining}")
        
        if remaining <= 0:
            log_auth_event(db, ip, payload.email, "lockout", "Verification locked out due to too many failed attempts.")
            raise HTTPException(
                status_code=403, 
                detail="Maximum verification attempts exceeded. Please request a new OTP code."
            )
        
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid verification code. You have {remaining} attempts remaining."
        )

    # OTP is correct: Activate User
    otp_session.is_verified = True
    user = db.query(User).filter(User.email == payload.email).first()
    if user:
        user.status = "active"
    db.commit()

    log_auth_event(db, ip, payload.email, "otp_verified", "Email successfully verified.")
    log_auth_event(db, ip, payload.email, "login_success", "User session created post-verification.")

    # Generate JWT Token and set both modern & legacy cookie names for full compatibility
    access_token = create_access_token(data={"sub": payload.email})
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=1800,
        expires=1800,
        samesite="strict",
        secure=False
    )
    response.set_cookie(
        key="sol_auth_token",
        value=access_token,
        httponly=False,
        max_age=1800,
        expires=1800,
        samesite="strict",
        secure=False
    )

    return {
        "status": "success",
        "message": "Account verified successfully.",
        "token": access_token,
        "redirect_url": "/app/dashboard"
    }

@router.post("/resend-otp")
async def resend_otp(
    payload: ResendOTPSchema,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    ip = request.client.host if request.client else "unknown"
    
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found.")
        
    if user.status == "active":
        raise HTTPException(status_code=400, detail="Account is already active. Please log in.")

    # Check resend cooldown
    otp_session = db.query(OTPSession).filter(OTPSession.email == payload.email).first()
    if otp_session:
        created_at = otp_session.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - created_at
        cooldown_limit = timedelta(seconds=OTP_RESEND_COOLDOWN_SECONDS)
        if elapsed < cooldown_limit:
            remaining = int((cooldown_limit - elapsed).total_seconds())
            log_auth_event(db, ip, payload.email, "resend_throttled", f"Resend requested too early. Remaining: {remaining}s")
            raise HTTPException(
                status_code=429, 
                detail=f"Please wait {remaining} seconds before requesting another code."
            )

    # Generate new OTP
    otp = generate_otp()
    hashed_code = hash_otp(otp)
    expire_time = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)

    if otp_session:
        otp_session.otp_hash = hashed_code
        otp_session.expires_at = expire_time
        otp_session.attempts = 0
        otp_session.is_verified = False
        otp_session.created_at = datetime.now(timezone.utc)
    else:
        otp_session = OTPSession(
            email=payload.email,
            otp_hash=hashed_code,
            expires_at=expire_time,
            attempts=0,
            is_verified=False
        )
        db.add(otp_session)
    db.commit()

    log_auth_event(db, ip, payload.email, "otp_sent", "New verification OTP generated and queued via resend.")
    background_tasks.add_task(email_worker.send_otp_email, payload.email, user.first_name or "User", otp)

    # Developer security debug logging
    print(f"\n[SECURITY DEBUG] Resent OTP for {payload.email}: {otp}\n", flush=True)
    try:
        with open("otp_debug.txt", "w") as f:
            f.write(f"Email: {payload.email}\nOTP: {otp}\nTimestamp: {datetime.now(timezone.utc)}\n")
    except Exception as e:
        print(f"[SECURITY DEBUG] Could not write to otp_debug.txt: {e}", flush=True)

    return {
        "status": "success",
        "message": "A new verification code has been sent to your email."
    }

@router.post("/login")
async def login(
    payload: LoginSchema,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    ip = request.client.host if request.client else "unknown"
    
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        log_auth_event(db, ip, payload.email, "login_failure", "Invalid email or password.")
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if user.status == "pending_verification":
        log_auth_event(db, ip, payload.email, "login_failure", "Account is inactive (pending verification).")
        raise HTTPException(
            status_code=403, 
            detail="Your account is not verified yet. Please verify your email first."
        )

    log_auth_event(db, ip, payload.email, "login_success", "User successfully authenticated.")

    if payload.remember:
        expires_delta = timedelta(days=30)
        max_age = 30 * 24 * 60 * 60
    else:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        max_age = ACCESS_TOKEN_EXPIRE_MINUTES * 60

    access_token = create_access_token(data={"sub": payload.email}, expires_delta=expires_delta)

    # Set both cookies for backward compatibility
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=max_age,
        expires=max_age,
        samesite="strict",
        secure=False
    )
    response.set_cookie(
        key="sol_auth_token",
        value=access_token,
        httponly=False,
        max_age=max_age,
        expires=max_age,
        samesite="strict",
        secure=False
    )

    return {
        "status": "success",
        "message": "Login successful.",
        "token": access_token,
        "redirect": "/app/dashboard"
    }

@router.post("/logout")
async def logout(response: Response, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "unknown"
    response.delete_cookie(key="access_token")
    response.delete_cookie(key="sol_auth_token")
    return {
        "status": "success",
        "message": "Logged out successfully.",
        "redirect_url": "/login"
    }

# --- OAuth Routes ---

@router.get("/google/login")
async def google_login(request: Request):
    redirect_uri = get_redirect_uri(request, "google")
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account"
    }
    authorization_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url=authorization_url)

@router.get("/google/callback", name="google_callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    from fastapi.responses import RedirectResponse
    ip = request.client.host if request.client else "unknown"
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    redirect_uri = get_redirect_uri(request, "google")
    
    async with httpx.AsyncClient() as client:
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }
        token_resp = await client.post(token_url, data=data)
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to retrieve Google token: {token_resp.text}")
            
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        
        userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        userinfo_resp = await client.get(userinfo_url, headers=headers)
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve Google user profile.")
            
        userinfo = userinfo_resp.json()
        email = userinfo.get("email")
        name = userinfo.get("name") or userinfo.get("given_name") or "GoogleUser"
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google account.")
            
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Create active user since OAuth verifies identity
            names = name.split(" ", 1)
            first_name = names[0] if names else "Google"
            last_name = names[1] if len(names) > 1 else "User"
            
            base_username = re.sub(r"[^a-zA-Z0-9_]", "", name.lower().replace(" ", "_"))
            if not base_username:
                base_username = email.split("@")[0]
            username = base_username
            counter = 1
            while db.query(User).filter(User.username == username).first():
                username = f"{base_username}_{counter}"
                counter += 1
                
            random_password = secrets.token_urlsafe(16)
            hashed_pwd = get_password_hash(random_password)
            user = User(
                first_name=first_name,
                last_name=last_name,
                username=username,
                email=email,
                hashed_password=hashed_pwd,
                status="active"
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            log_auth_event(db, ip, email, "register", "User registered automatically via Google OAuth (active).")
        else:
            if user.status != "active":
                user.status = "active"
                db.commit()
                
        log_auth_event(db, ip, email, "login_success", "User successfully authenticated via Google OAuth.")
        jwt_token = create_access_token(data={"sub": email})
        
        resp = RedirectResponse(url="/app/dashboard")
        resp.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            max_age=1800,
            expires=1800,
            samesite="strict",
            secure=False
        )
        resp.set_cookie(
            key="sol_auth_token",
            value=jwt_token,
            httponly=False,
            max_age=1800,
            expires=1800,
            samesite="strict",
            secure=False
        )
        return resp

@router.get("/github/login")
async def github_login(request: Request):
    redirect_uri = get_redirect_uri(request, "github")
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "user:email"
    }
    authorization_url = "https://github.com/login/oauth/authorize?" + urllib.parse.urlencode(params)
    return RedirectResponse(url=authorization_url)

@router.get("/github/callback", name="github_callback")
async def github_callback(request: Request, db: Session = Depends(get_db)):
    from fastapi.responses import RedirectResponse
    ip = request.client.host if request.client else "unknown"
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    redirect_uri = get_redirect_uri(request, "github")
    
    async with httpx.AsyncClient() as client:
        token_url = "https://github.com/login/oauth/access_token"
        headers = {"Accept": "application/json"}
        data = {
            "code": code,
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "redirect_uri": redirect_uri
        }
        token_resp = await client.post(token_url, data=data, headers=headers)
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve GitHub token.")
            
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="Access token not returned by GitHub.")
            
        user_url = "https://api.github.com/user"
        headers = {
            "Authorization": f"token {access_token}",
            "User-Agent": "FastAPI-OAuth"
        }
        user_resp = await client.get(user_url, headers=headers)
        if user_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve GitHub user profile.")
            
        user_data = user_resp.json()
        github_username = user_data.get("login")
        name = user_data.get("name") or github_username or "GitHubUser"
        email = user_data.get("email")
        
        if not email:
            emails_url = "https://api.github.com/user/emails"
            emails_resp = await client.get(emails_url, headers=headers)
            if emails_resp.status_code == 200:
                emails_list = emails_resp.json()
                for email_info in emails_list:
                    if email_info.get("primary") and email_info.get("verified"):
                        email = email_info.get("email")
                        break
                if not email:
                    for email_info in emails_list:
                        if email_info.get("verified"):
                            email = email_info.get("email")
                            break
                            
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by GitHub account.")
            
        user = db.query(User).filter(User.email == email).first()
        if not user:
            names = name.split(" ", 1)
            first_name = names[0] if names else "GitHub"
            last_name = names[1] if len(names) > 1 else "User"
            
            base_username = re.sub(r"[^a-zA-Z0-9_]", "", github_username.lower())
            if not base_username:
                base_username = email.split("@")[0]
            username = base_username
            counter = 1
            while db.query(User).filter(User.username == username).first():
                username = f"{base_username}_{counter}"
                counter += 1
                
            random_password = secrets.token_urlsafe(16)
            hashed_pwd = get_password_hash(random_password)
            user = User(
                first_name=first_name,
                last_name=last_name,
                username=username,
                email=email,
                hashed_password=hashed_pwd,
                status="active"
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            log_auth_event(db, ip, email, "register", "User registered automatically via GitHub OAuth (active).")
        else:
            if user.status != "active":
                user.status = "active"
                db.commit()
                
        log_auth_event(db, ip, email, "login_success", "User successfully authenticated via GitHub OAuth.")
        jwt_token = create_access_token(data={"sub": email})
        
        resp = RedirectResponse(url="/app/dashboard")
        resp.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            max_age=1800,
            expires=1800,
            samesite="strict",
            secure=False
        )
        resp.set_cookie(
            key="sol_auth_token",
            value=jwt_token,
            httponly=False,
            max_age=1800,
            expires=1800,
            samesite="strict",
            secure=False
        )
        return resp
