import os
import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Union, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from pydantic import BaseModel
from dotenv import load_dotenv

from .database import get_db
from .models import User

# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "sol_super_secret_jwt_key_2026")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080")) # Default 7 days

# Dual scheme support (bcrypt for new, pbkdf2_sha256 for existing users)
pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")

class OAuth2PasswordBearerWithCookie(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            return request.cookies.get("sol_auth_token") or request.cookies.get("access_token")
        scheme, _, param = authorization.partition(" ")
        if not authorization or scheme.lower() != "bearer":
            return request.cookies.get("sol_auth_token") or request.cookies.get("access_token")
        return param

oauth2_scheme = OAuth2PasswordBearerWithCookie(tokenUrl="/api/auth/login")

class TokenData(BaseModel):
    sub: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# OTP and Auth utility functions
def generate_otp() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(6))

def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        if sub is None:
            raise credentials_exception
        token_data = TokenData(sub=sub)
    except JWTError:
        raise credentials_exception
        
    # Check if subject is email or user ID
    if "@" in token_data.sub:
        user = db.query(User).filter(User.email == token_data.sub).first()
    else:
        try:
            user_id = int(token_data.sub)
            user = db.query(User).filter(User.id == user_id).first()
        except ValueError:
            user = None
            
    if user is None:
        raise credentials_exception
        
    # Auto-promote administrators on verification/login
    if user.email in ('solixagentic@gmail.com', 'socilaw715@luxudata.com', 'heizul@itmo.edu.pl') and not getattr(user, 'is_admin', False):
        user.is_admin = True
        db.commit()
        
    # Check active status
    if hasattr(user, "status") and user.status == "pending_verification":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not verified yet. Please verify your email first."
        )
        
    from backend.store import active_user_id
    active_user_id.set(user.id)
    return user

def get_optional_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        if sub is None:
            return None
        if "@" in sub:
            user = db.query(User).filter(User.email == sub).first()
        else:
            user_id = int(sub)
            user = db.query(User).filter(User.id == user_id).first()
        
        if user:
            from backend.store import active_user_id
            active_user_id.set(user.id)
            return user
    except Exception:
        pass
    return None
