import sys
import os
import uuid
import warnings
import io
import json
import random
import string
import time

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
#  Path wiring — lets us import from the sibling run/ tree
# ─────────────────────────────────────────────────────────
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_run_dir = os.path.dirname(_backend_dir)          # …/run/
sys.path.insert(0, _run_dir)                       # gives access to core/, utils/, data_layer/
sys.path.insert(0, _backend_dir)                   # gives access to local backend modules like database_utils

# Suppress noisy warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
pd.set_option("future.no_silent_downcasting", True)

# ─────────────────────────────────────────────────────────
#  Core engine imports
# ─────────────────────────────────────────────────────────
from core.analyzer import MetadataAnalyzer
from core.cleaner import SmartDataCleaner
from utils.ai_imputer import AIImputer
from backend.utils.kaggle_orchestrator import KaggleStudioOrchestrator

# ─────────────────────────────────────────────────────────
#  FastAPI
# ─────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request, Depends
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
import uvicorn

from backend.database import engine, Base, get_db, SessionLocal
from backend.models import User, Notification, JobRecord, Project, Task, TaskRun, Feedback, UserSettings, DashboardPreferences

# ─────────────────────────────────────────────────────────
#  Environment variables dynamic multi-tenancy patching
# ─────────────────────────────────────────────────────────
import os
from backend.store import active_user_id

_original_getenv = os.getenv
_original_environ_getitem = os.environ.__class__.__getitem__
_original_environ_contains = os.environ.__class__.__contains__

def is_placeholder(val: str) -> bool:
    if not val:
        return True
    val_lower = val.lower().strip()
    placeholders = [
        "your_api_key",
        "placeholder",
        "gsk_xpiqz",
        "gsk_6k5y7",
        "gsk_1myfi",
    ]
    if len(val_lower) < 15:
        return True
    for p in placeholders:
        if p in val_lower:
            return True
    return False

def patched_getenv(key, default=None):
    mapped_keys = {
        "KAGGLE_USERNAME": "kaggle_username",
        "KAGGLE_KEY": "kaggle_key",
        "KAGGLE_API_TOKEN": "kaggle_key",
        "GROQ_API_KEY": "groq_api_key",
        "ELEVENLABS_API_KEY": "elevenlabs_api_key",
        "ELEVENLABS_ID": "elevenlabs_id"
    }
    if key in mapped_keys:
        uid = active_user_id.get()
        if uid is not None:
            from backend.database import SessionLocal
            from backend.models import UserSettings
            db = SessionLocal()
            try:
                settings = db.query(UserSettings).filter_by(user_id=uid).first()
                if settings:
                    val = getattr(settings, mapped_keys[key], None)
                    if val is not None and val.strip() != "" and not is_placeholder(val):
                        return val
            except Exception:
                pass
            finally:
                db.close()
            return default
    return _original_getenv(key, default)

def patched_environ_getitem(self, key):
    mapped_keys = {
        "KAGGLE_USERNAME": "kaggle_username",
        "KAGGLE_KEY": "kaggle_key",
        "KAGGLE_API_TOKEN": "kaggle_key",
        "GROQ_API_KEY": "groq_api_key",
        "ELEVENLABS_API_KEY": "elevenlabs_api_key",
        "ELEVENLABS_ID": "elevenlabs_id"
    }
    if key in mapped_keys:
        uid = active_user_id.get()
        if uid is not None:
            from backend.database import SessionLocal
            from backend.models import UserSettings
            db = SessionLocal()
            try:
                settings = db.query(UserSettings).filter_by(user_id=uid).first()
                if settings:
                    val = getattr(settings, mapped_keys[key], None)
                    if val is not None and val.strip() != "" and not is_placeholder(val):
                        return val
            except Exception:
                pass
            finally:
                db.close()
            raise KeyError(key)
    return _original_environ_getitem(self, key)

def patched_environ_get(key, default=None):
    try:
        return os.environ[key]
    except KeyError:
        return default

def patched_environ_contains(self, key):
    mapped_keys = {
        "KAGGLE_USERNAME": "kaggle_username",
        "KAGGLE_KEY": "kaggle_key",
        "KAGGLE_API_TOKEN": "kaggle_key",
        "GROQ_API_KEY": "groq_api_key",
        "ELEVENLABS_API_KEY": "elevenlabs_api_key",
        "ELEVENLABS_ID": "elevenlabs_id"
    }
    if key in mapped_keys:
        uid = active_user_id.get()
        if uid is not None:
            from backend.database import SessionLocal
            from backend.models import UserSettings
            db = SessionLocal()
            try:
                settings = db.query(UserSettings).filter_by(user_id=uid).first()
                if settings:
                    val = getattr(settings, mapped_keys[key], None)
                    if val is not None and val.strip() != "" and not is_placeholder(val):
                        return True
            except Exception:
                pass
            finally:
                db.close()
            return False
    return _original_environ_contains(self, key)

os.getenv = patched_getenv
os.environ.__class__.__getitem__ = patched_environ_getitem
os.environ.get = patched_environ_get
os.environ.__class__.__contains__ = patched_environ_contains

from backend.tools.ocr.router import router as ocr_router
from backend.tools.forms.router import router as form_router
from backend.tools.semantic_mapper.router import router as semantic_router
from backend.tools.ml_advisor.router import router as ml_router
from backend.tools.audit.router import router as audit_router
from backend.tools.data_noise.router import router as data_noise_router
from backend.tools.narrator.router import router as narrator_router
from backend.tools.viz_engine.router import router as viz_router
from backend.tools.viz_engine.engine import VizEngine
from backend.tools.copilot.router import router as copilot_router
from backend.tools.automl.router import router as automl_router
from backend.auth_routes import router as auth_routes_router
from backend.tools.dataset_advisor.router import dataset_advisor_router
from backend.tools.synthetic_data.router import router as synthetic_router
from backend.tools.dashboard.router import router as dashboard_router
from backend.tools.chatbot.router import router as chatbot_router
from backend.tools.admin.router import router as admin_router

from backend.middleware.rate_limit import SimpleRateLimiter
from starlette.middleware.sessions import SessionMiddleware

from backend.store import (
    _store, _store_parquet_path, _store_tasks, _store_ext, _store_filename, _store_goals, _store_is_db,
    _audit_store, _viz_store, _discovery_store
)
from backend.tools.audit.engine import AuditReportBuilder
from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user

# Create database tables
Base.metadata.create_all(bind=engine)

try:
    from backend.migrate_db import migrate
    migrate()
except Exception as e:
    print(f"Database migration failed: {e}")

app = FastAPI(
    title="SOL Agent — Enterprise Data Factory",
    version="V1.0",
    description="High-performance AI data cleaning API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "b91a603957beab8d956f2f9f98f6d89bdfad741df747372cf91c0e358b68832a")
)
app.add_middleware(SimpleRateLimiter, requests_limit=30, window_seconds=60)

frontend_dir = os.path.join(_run_dir, "frontend")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/team-images", StaticFiles(directory=os.path.join(_run_dir, "صور التيم")), name="team-images")

# Mount Routers
app.include_router(auth_routes_router)
app.include_router(ocr_router)
app.include_router(form_router)
app.include_router(semantic_router)
app.include_router(ml_router)
app.include_router(audit_router)
app.include_router(data_noise_router)
app.include_router(narrator_router)
app.include_router(viz_router)
app.include_router(copilot_router)
app.include_router(automl_router)
app.include_router(dataset_advisor_router)
app.include_router(synthetic_router)
app.include_router(dashboard_router)
app.include_router(chatbot_router)
app.include_router(admin_router)

def add_locale_context(request: Request):
    locale = request.cookies.get("sol_locale", "en")
    if locale not in ("en", "ar"):
        locale = "en"
    return {
        "locale": locale,
        "locale_dir": "rtl" if locale == "ar" else "ltr",
        "locale_lang": locale
    }

templates = Jinja2Templates(
    directory=os.path.join(frontend_dir, "templates"),
    context_processors=[add_locale_context]
)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    # Automatically redirect unauthenticated HTML requests to login
    if exc.status_code == 401 and request.url.path.startswith("/app"):
        return RedirectResponse(url="/login")
    if exc.status_code == 403 and "verify" in str(exc.detail).lower() and request.url.path.startswith("/app"):
        return RedirectResponse(url="/login?error=verification_pending")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# store is imported from backend.store

analyzer = MetadataAnalyzer()

# ─────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────

ERROR_TOKENS = {"ERROR", "error", "UNKNOWN", "unknown", "?", "-", "Not Started", "Null", "NULL", "N/A", "n/a", "na", "NA"}


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-serialisable records (NaN → None)."""
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", date_format="iso"))


def _build_diff_map(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, list[int]]:
    """
    Return {column_name: [row_indices_that_changed]}.
    A cell is 'changed' if the after value differs from the before value
    (treats NaN == NaN as equal so we only flag actual fills/changes).
    """
    diff_map: dict[str, list[int]] = {}
    shared_cols = [c for c in before.columns if c in after.columns]
    for col in shared_cols:
        changed_rows: list[int] = []
        for idx in range(min(len(before), len(after))):
            b_val = before[col].iloc[idx]
            a_val = after[col].iloc[idx]
            b_nan = pd.isna(b_val) if not isinstance(b_val, str) else (str(b_val) in ERROR_TOKENS)
            a_nan = pd.isna(a_val) if not isinstance(a_val, str) else (str(a_val) in ERROR_TOKENS)
            if b_nan and not a_nan:
                changed_rows.append(idx)
            elif not b_nan and not a_nan and b_val != a_val:
                changed_rows.append(idx)
        if changed_rows:
            diff_map[col] = changed_rows
    return diff_map


def _is_nan_or_error(val) -> bool:
    """True when a cell value is considered dirty (NaN or known error token)."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() in ERROR_TOKENS:
        return True
    return False


def _build_nan_map(df: pd.DataFrame) -> dict[str, list[int]]:
    """Return {column_name: [row_indices_that_are_NaN/Error]} for the BEFORE table."""
    nan_map: dict[str, list[int]] = {}
    for col in df.columns:
        bad_rows = [idx for idx, val in enumerate(df[col]) if _is_nan_or_error(val)]
        if bad_rows:
            nan_map[col] = bad_rows
    return nan_map


# ─────────────────────────────────────────────────────────
#  Strategy builders (deterministic — no LLM call)
# ─────────────────────────────────────────────────────────

def _build_strategy(level: str, metadata: dict, goal: str = None) -> dict:
    """
    Build a cleaning strategy_json compatible with SmartDataCleaner.execute_strategy().
    level: 'alpha' | 'beta' | 'gamma'
    """
    cols = metadata.get("columns_info", [])
    plan: dict[str, str] = {}
    remove_dups = level in ("beta", "gamma")
    summaries = []

    goal_injected = False
    if goal and goal.strip():
        # Mocking the AI incorporating the user's natural language goal into specific actions.
        plan["__GLOBAL_GOAL__"] = f"Goal constraints analyzed: {goal}"
        goal_injected = True

    if remove_dups:
        summaries.append("Remove exact duplicate rows")

    for col in cols:
        name = col["name"]
        semantic = col.get("semantic_type", "")
        dtype = col.get("physical_type", "").lower()
        missing = col.get("missing_count", 0)
        col_lower = name.lower()
        is_sensitive = col.get("is_sensitive", False)

        # Skip sensitive columns for gamma (no aggressive drops)
        if is_sensitive and level == "gamma":
            if missing > 0:
                plan[name] = "smart_impute"
                summaries.append(f"Impute missing values in sensitive col: '{name}'")
            continue

        # Date columns
        if "date" in semantic.lower() or "time" in semantic.lower() or "date" in col_lower:
            plan[name] = "standardize_date"
            summaries.append(f"Standardize formatting for date/time '{name}'")
            continue

        # Numeric columns
        if dtype in ("int64", "float64"):
            is_id_like = any(x in col_lower for x in ["id", "phone", "mobile", "code", "zip", "year"])
            if is_id_like:
                if missing > 0:
                    plan[name] = "smart_impute"
                    summaries.append(f"Safe impute ID-like col: '{name}'")
            else:
                if level == "alpha":
                    if missing > 0:
                        plan[name] = "smart_impute"
                        summaries.append(f"Impute missing numeric in '{name}' using Median")
                elif level == "beta":
                    plan[name] = "remove_outliers"
                    summaries.append(f"Impute missing & Z-score Outlier Removal for '{name}'")
                else:  # gamma
                    plan[name] = "remove_outliers"
                    summaries.append(f"Aggressive Outlier Removal & Impute for '{name}'")
            continue

        # Text columns — fuzzy fix for known domains
        if dtype == "object":
            text_fix_keywords = ["city", "job", "title", "country", "name", "governorate", "department", "status", "category"]
            numeric_text_keywords = ["salary", "price", "amount", "cost", "budget", "revenue", "income"]

            if any(k in col_lower for k in numeric_text_keywords):
                plan[name] = "remove_outliers"
                summaries.append(f"Clean numeric text and trim outliers in '{name}'")
            elif any(k in col_lower for k in text_fix_keywords):
                plan[name] = "fuzzy_fix"
                summaries.append(f"Fuzzy match & normalize text in '{name}'")
            elif missing > 0:
                plan[name] = "smart_impute"
                summaries.append(f"Fill missing text in '{name}' with mode/placeholder")

    return {
        "remove_duplicates": remove_dups,
        "cleaning_strategy": plan,
        "summaries": summaries,
        "goal_injected": goal_injected
    }


# ─────────────────────────────────────────────────────────
#  Frontend Routes
# ─────────────────────────────────────────────────────────

@app.get("/authorize/google")
async def legacy_google_callback(request: Request):
    query_params = request.url.query
    target_url = f"/api/v1/auth/google/callback?{query_params}" if query_params else "/api/v1/auth/google/callback"
    return RedirectResponse(url=target_url)

@app.get("/authorize/github")
async def legacy_github_callback(request: Request):
    query_params = request.url.query
    target_url = f"/api/v1/auth/github/callback?{query_params}" if query_params else "/api/v1/auth/github/callback"
    return RedirectResponse(url=target_url)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("public/landing.html", {"request": request})

@app.get("/pricing", response_class=HTMLResponse)
def read_pricing(request: Request):
    return templates.TemplateResponse("public/pricing.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def read_login(request: Request):
    return templates.TemplateResponse("public/login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def read_register(request: Request):
    return templates.TemplateResponse("public/register.html", {"request": request})

@app.get("/verify-otp", response_class=HTMLResponse)
def read_verify_otp(request: Request, email: str = ""):
    return templates.TemplateResponse("public/verify_otp.html", {"request": request, "email": email})

@app.get("/forgot-password", response_class=HTMLResponse)
def read_forgot_password(request: Request):
    return templates.TemplateResponse("public/forgot_password.html", {"request": request})

@app.get("/reset-password", response_class=HTMLResponse)
def read_reset_password(request: Request):
    return templates.TemplateResponse("public/reset_password.html", {"request": request})

@app.get("/app/dashboard", response_class=HTMLResponse)
def read_dashboard(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/dashboard.html", {"request": request})

@app.get("/app/admin", response_class=HTMLResponse)
def read_admin_dashboard(request: Request, current_user: User = Depends(get_current_user)):
    if not getattr(current_user, "is_admin", False):
        return RedirectResponse(url="/app/dashboard")
    return templates.TemplateResponse("app/admin_dashboard.html", {"request": request})

@app.get("/app/ocr", response_class=HTMLResponse)
async def read_ocr_engine(request: Request, current_user = Depends(get_current_user)):
    return templates.TemplateResponse("app/ocr_engine.html", {"request": request})

@app.get("/app/studio", response_class=HTMLResponse)
def read_studio(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/cleaning_studio.html", {"request": request})

@app.get("/app/forms", response_class=HTMLResponse)
def read_forms_dashboard(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/forms_dashboard.html", {"request": request})

@app.get("/app/semantic-mapper", response_class=HTMLResponse)
def read_semantic_mapper(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/semantic_mapper.html", {"request": request})

@app.get("/app/ml-advisor", response_class=HTMLResponse)
def read_ml_advisor(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/ml_advisor.html", {"request": request})

@app.get("/app/dataset-advisor", response_class=HTMLResponse)
def read_dataset_advisor(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/dataset_advisor.html", {"request": request})

@app.on_event("startup")
async def startup_dataset_advisor():
    print("Startup: Initializing Dataset Advisor database and vectors...")
    try:
        from backend.tools.dataset_advisor.db.session import init_db, async_session_factory
        from backend.tools.dataset_advisor.models.database import Dataset
        from backend.tools.dataset_advisor.services.ingestion.orchestrator import ingest_orchestrator
        from sqlalchemy.future import select
        from sqlalchemy import func
        
        await init_db()
        
        # Check if DB has seeds, if empty automatically seed to ensure instant out-of-the-box runtime
        async with async_session_factory() as session:
            stmt = select(func.count()).select_from(Dataset)
            result = await session.execute(stmt)
            count = result.scalar_one()
            
            # Check if any dataset is missing file_size (pre-migration)
            stmt_null = select(func.count()).select_from(Dataset).where(Dataset.file_size == None)
            result_null = await session.execute(stmt_null)
            null_count = result_null.scalar_one()
            
            if count == 0 or null_count > 0:
                print("Database contains 0 datasets or old datasets with missing file size. Triggering clean seeding...")
                if null_count > 0:
                    from sqlalchemy import delete
                    await session.execute(delete(Dataset))
                    await session.commit()
                await ingest_orchestrator.ingest_all_seeds(session)
            else:
                print(f"Database initialized with {count} pre-existing datasets. Skipping auto-seeding.")
    except Exception as e:
        print(f"Failed during startup dataset advisor database check/seeding: {e}")






@app.get("/app/data-noise", response_class=HTMLResponse)
def read_data_noise(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/data_noise.html", {"request": request})

@app.get("/app/audit", response_class=HTMLResponse)
def read_audit_report(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/audit_report.html", {"request": request})

@app.get("/f/{form_id}", response_class=HTMLResponse)
def read_form_fill(request: Request, form_id: int):
    return templates.TemplateResponse("public/form_fill.html", {"request": request, "form_id": form_id})

@app.get("/app/settings", response_class=HTMLResponse)
def read_settings(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/settings.html", {"request": request})

@app.get("/solutions", response_class=HTMLResponse)
def read_solutions(request: Request):
    return templates.TemplateResponse("public/solutions.html", {"request": request})

@app.get("/about-team", response_class=HTMLResponse)
def read_about_team(request: Request):
    return templates.TemplateResponse("public/about_team.html", {"request": request})


import html
import re

@app.get("/feedback", response_class=HTMLResponse)
def read_feedback(request: Request):
    return templates.TemplateResponse("public/feedback.html", {"request": request})

@app.post("/api/feedback")
def submit_feedback(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None),
    message: str = Form(...),
    db: Session = Depends(get_db)
):
    name = name.strip()
    email = email.strip()
    message = message.strip()
    
    if not name or not email or not message:
        raise HTTPException(status_code=400, detail="All required fields must be filled")
        
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise HTTPException(status_code=400, detail="Invalid email format")
        
    safe_name = html.escape(name)
    safe_email = html.escape(email)
    safe_phone = html.escape(phone) if phone else None
    safe_message = html.escape(message)
    
    new_feedback = Feedback(
        name=safe_name,
        email=safe_email,
        phone=safe_phone,
        message=safe_message
    )
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    
    return {"status": "success", "message": "Feedback submitted successfully"}

@app.get("/feedback-admin-portal-solix", response_class=HTMLResponse)
def get_feedback_admin(request: Request, db: Session = Depends(get_db)):
    response = templates.TemplateResponse(
        "public/feedback_admin.html", 
        {"request": request, "feedbacks": None, "authenticated": False}
    )
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    
    auth_cookie = request.cookies.get("solix_admin_auth")
    if auth_cookie == "10mohamed10":
        feedbacks = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
        response = templates.TemplateResponse(
            "public/feedback_admin.html", 
            {"request": request, "feedbacks": feedbacks, "authenticated": True}
        )
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response
        
    return response

@app.post("/feedback-admin-portal-solix/login")
def login_feedback_admin(password: str = Form(...)):
    if password == "10mohamed10":
        response = RedirectResponse(url="/feedback-admin-portal-solix", status_code=303)
        response.set_cookie(
            "solix_admin_auth", 
            "10mohamed10", 
            httponly=True, 
            max_age=86400, 
            samesite="lax"
        )
        return response
    return RedirectResponse(url="/feedback-admin-portal-solix?error=1", status_code=303)

@app.post("/feedback-admin-portal-solix/logout")
def logout_feedback_admin():
    response = RedirectResponse(url="/feedback-admin-portal-solix", status_code=303)
    response.delete_cookie("solix_admin_auth")
    return response

@app.post("/feedback-admin-portal-solix/delete/{feedback_id}")
def delete_feedback(feedback_id: int, request: Request, db: Session = Depends(get_db)):
    auth_cookie = request.cookies.get("solix_admin_auth")
    if auth_cookie != "10mohamed10":
        raise HTTPException(status_code=403, detail="Not authorized")
        
    fb_item = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb_item:
        raise HTTPException(status_code=404, detail="Feedback not found")
        
    db.delete(fb_item)
    db.commit()
    return {"status": "success", "message": "Feedback deleted successfully"}

@app.get("/docs", response_class=HTMLResponse)
def read_docs(request: Request):
    return RedirectResponse(url="/about-team")

@app.get("/architecture", response_class=HTMLResponse)
def read_architecture(request: Request):
    return RedirectResponse(url="/solutions")

@app.get("/status", response_class=HTMLResponse)
def read_status(request: Request):
    return templates.TemplateResponse("public/status.html", {"request": request})

@app.get("/privacy", response_class=HTMLResponse)
def read_privacy(request: Request):
    return templates.TemplateResponse("public/_legal_layout.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
def read_terms(request: Request):
    return templates.TemplateResponse("public/_legal_layout.html", {"request": request})

@app.get("/app", response_class=HTMLResponse)
def read_app(request: Request, user: User = Depends(get_current_user)):
    return RedirectResponse(url="/app/dashboard")

@app.get("/app/viz-report", response_class=HTMLResponse)
def read_viz_report(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/viz_report.html", {"request": request})

@app.get("/app/chat-with-data", response_class=HTMLResponse)
def read_chat_with_data(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/chat_with_data.html", {"request": request})

@app.get("/app/automl-studio", response_class=HTMLResponse)
def read_automl_studio(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/automl_studio.html", {"request": request})

@app.get("/app/synthetic-studio", response_class=HTMLResponse)
def read_synthetic_studio(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/synthetic_studio.html", {"request": request})

@app.get("/app/temp-login")
def temp_login():
    from fastapi.responses import RedirectResponse
    from backend.auth import create_access_token
    token = create_access_token({"sub": "test@test.com"})
    response = RedirectResponse(url="/app/synthetic-studio")
    response.set_cookie(key="sol_auth_token", value=token, httponly=True)
    return response



# ─────────────────────────────────────────────────────────
#  Auth API Profile Endpoints (Unified settings support)
# ─────────────────────────────────────────────────────────

from pydantic import BaseModel, EmailStr
from typing import Optional

class ProfileUpdateSchema(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    job_title: Optional[str] = None
    organization: Optional[str] = None

class ChangePasswordSchema(BaseModel):
    old_password: str
    new_password: str

@app.get("/api/auth/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "email": current_user.email,
        "avatar_url": getattr(current_user, "avatar_url", None),
        "is_admin": getattr(current_user, "is_admin", False)
    }

@app.get("/api/me")
def read_users_me_v2(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "username": current_user.username,
        "email": current_user.email,
        "status": current_user.status,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "job_title": current_user.job_title,
        "organization": current_user.organization,
        "avatar_url": getattr(current_user, "avatar_url", None)
    }

@app.put("/api/me")
def update_user_profile(
    payload: ProfileUpdateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if email is already taken by another user
    if payload.email != current_user.email:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use.")
        current_user.email = payload.email

    current_user.first_name = payload.first_name
    current_user.last_name = payload.last_name
    current_user.job_title = payload.job_title
    current_user.organization = payload.organization
    
    db.commit()
    db.refresh(current_user)
    return {"status": "success"}

@app.post("/api/change-password")
def change_user_password(
    payload: ChangePasswordSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not verify_password(payload.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password.")
    
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")

    current_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    return {"status": "success"}

@app.post("/api/settings/upload-avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    avatars_dir = os.path.join(_run_dir, "frontend", "static", "avatars")
    os.makedirs(avatars_dir, exist_ok=True)
    
    filename = f"user_{current_user.id}.png"
    file_path = os.path.join(avatars_dir, filename)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save avatar: {str(e)}")
        
    avatar_url = f"/static/avatars/{filename}?t=" + str(int(time.time()))
    current_user.avatar_url = avatar_url
    db.commit()
    
    return {"status": "success", "avatar_url": avatar_url}

@app.post("/api/settings/delete-avatar")
def delete_avatar(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.avatar_url:
        # Strip query parameters if present
        clean_url = current_user.avatar_url.split('?')[0]
        filename = os.path.basename(clean_url)
        avatars_dir = os.path.join(_run_dir, "frontend", "static", "avatars")
        file_path = os.path.join(avatars_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing avatar file: {e}")
                
    current_user.avatar_url = None
    db.commit()
    return {"status": "success"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """
    Accept CSV or XLSX/XLS, analyse it, store the raw DataFrame (or preview for large datasets), return a data profile.
    """
    import polars as pl
    import shutil
    
    filename = file.filename or "upload.csv"
    ext = os.path.splitext(filename)[1].lower()

    supported_exts = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    if ext not in supported_exts:
        raise HTTPException(status_code=400, detail=f"Format {ext} is not supported yet.")

    dataset_id = str(uuid.uuid4())
    temp_dir = "temp_snapshots"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded file in chunks (memory-safe)
    temp_raw_path = os.path.join(temp_dir, f"{dataset_id}_raw{ext}")
    with open(temp_raw_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            buffer.write(chunk)

    try:
        # Analyze file lazily using Polars via the updated analyzer
        metadata = analyzer.analyze_file(filename=filename, file_path=temp_raw_path)
    except ValueError as e:
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
        raise HTTPException(status_code=422, detail=str(e))

    parquet_path = os.path.join(temp_dir, f"{dataset_id}.parquet")

    # Convert/copy to Parquet
    try:
        if ext == ".parquet":
            shutil.copy2(temp_raw_path, parquet_path)
        elif ext == ".csv":
            try:
                # Out-of-core streaming conversion
                pl.scan_csv(temp_raw_path, infer_schema_length=10000).sink_parquet(parquet_path)
            except Exception:
                # Fallback to standard read/write
                df_temp = pd.read_csv(temp_raw_path)
                df_temp.to_parquet(parquet_path, index=False)
        elif ext in (".xlsx", ".xls"):
            df_temp = pd.read_excel(temp_raw_path)
            df_temp.to_parquet(parquet_path, index=False)
        elif ext == ".json":
            df_temp = pd.read_json(temp_raw_path)
            df_temp.to_parquet(parquet_path, index=False)
        else:
            df_temp = pd.read_csv(temp_raw_path)
            df_temp.to_parquet(parquet_path, index=False)
    except Exception as e:
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
        raise HTTPException(status_code=422, detail=f"Failed to convert dataset to Parquet: {e}")

    # Remove temporary raw upload
    if os.path.exists(temp_raw_path):
        os.remove(temp_raw_path)

    # Determine shape and load into store
    row_count = metadata["rows"]
    
    # Store cleanups
    _store.clear()
    _store_ext.clear()
    _store_filename.clear()
    _store_goals.clear()
    _store_is_db.clear()
    _audit_store.clear()
    _discovery_store.clear()
    _viz_store.clear()
    _store_parquet_path.clear()

    # If small (<= 50,000 rows), load fully in pandas to guarantee 100% backward compatibility
    # Else, load only the 500-row preview to prevent OOM
    if row_count <= 50000:
        df = pl.read_parquet(parquet_path).to_pandas()
    else:
        df = pl.read_parquet(parquet_path, n_rows=500).to_pandas()

    _store[dataset_id] = df
    _store_parquet_path[dataset_id] = parquet_path
    _store_ext[dataset_id] = ext.lower()
    _store_filename[dataset_id] = filename

    # Build nan_map
    nan_map = _build_nan_map(df)

    # ── VizEngine MODE_DISCOVERY — build discovery charts right after upload ──
    try:
        viz = VizEngine(raw_df=df)
        _discovery_store[dataset_id] = viz.discovery()
    except Exception as _viz_err:
        print(f"[!] VizEngine discovery error (non-blocking): {_viz_err}")
    # ─────────────────────────────────────────────────────────────────────────

    return JSONResponse({
        "dataset_id": dataset_id,
        "metadata": metadata,
        "nan_map": nan_map,
        "raw_data": _df_to_records(df.head(500)),
    })


@app.get("/api/metadata/{dataset_id}")
def get_dataset_metadata(dataset_id: str, current_user: User = Depends(get_current_user)):
    """Retrieve metadata of the uploaded dataset by running the MetadataAnalyzer lazily on the stored Parquet file."""
    parquet_path = _store_parquet_path.get(dataset_id)
    if not parquet_path:
        # Check if preview exists or fallback to _store to find the dataset key
        if dataset_id not in _store:
            raise HTTPException(status_code=404, detail="Dataset not found. Please upload it first.")
        # If it's in store but parquet path is missing, generate one
        temp_dir = "temp_snapshots"
        os.makedirs(temp_dir, exist_ok=True)
        parquet_path = os.path.join(temp_dir, f"{dataset_id}.parquet")
        _store[dataset_id].to_parquet(parquet_path, index=False)
        _store_parquet_path[dataset_id] = parquet_path

    filename = _store_filename.get(dataset_id, "dataset.csv")
    try:
        meta = analyzer.analyze_file(filename=filename, file_path=parquet_path)
        return JSONResponse(meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





class PreviewRequest(BaseModel):
    metadata: dict
    goal: str = None

@app.post("/api/strategies/preview")
def preview_strategies(req: PreviewRequest, current_user: User = Depends(get_current_user)):
    """Dynamically build cleaning strategy plans to display in the frontend without running them."""
    return {
        "alpha": _build_strategy("alpha", req.metadata, req.goal),
        "beta": _build_strategy("beta", req.metadata, req.goal),
        "gamma": _build_strategy("gamma", req.metadata, req.goal),
    }

def run_background_clean(
    task_id: str,
    dataset_id: str,
    strategy_level: str,
    goal: str | None,
    approved_actions: str | None,
):
    try:
        _store_tasks[task_id]["status"] = "processing"
        _store_tasks[task_id]["progress"] = 10
        
        parquet_path = _store_parquet_path.get(dataset_id)
        if not parquet_path:
            if dataset_id not in _store:
                raise Exception("Dataset not found.")
            temp_dir = "temp_snapshots"
            os.makedirs(temp_dir, exist_ok=True)
            parquet_path = os.path.join(temp_dir, f"{dataset_id}.parquet")
            _store[dataset_id].to_parquet(parquet_path, index=False)
            _store_parquet_path[dataset_id] = parquet_path
            
        _store_tasks[task_id]["progress"] = 25
        
        # Load dataset (Safe on OS/FastAPI memory limits by tuning)
        raw_df = pd.read_parquet(parquet_path)
        _store_tasks[task_id]["progress"] = 40
        
        # Build metadata for strategy planner
        try:
            metadata = analyzer.analyze_file(filename=_store_filename.get(dataset_id, "data.parquet"), file_path=parquet_path)
        except Exception:
            metadata = {"columns_info": []}
            
        _store_tasks[task_id]["progress"] = 50
        
        strategy_json = _build_strategy(strategy_level, metadata, goal)
        
        policy_config = None
        if approved_actions:
            try:
                approved_list = json.loads(approved_actions)
                if isinstance(approved_list, list):
                    policy_config = {"approved_actions": approved_list}
            except Exception as e:
                print(f"[!] Failed to parse approved_actions: {e}")
                
        _store_tasks[task_id]["progress"] = 65
        
        # Execute cleaning
        cleaner = SmartDataCleaner(raw_df, policy_config=policy_config)
        result = cleaner.execute_strategy(strategy_json)
        cleaned_df, report = result if isinstance(result, tuple) else (result, {"actions": []})
        
        _store_tasks[task_id]["progress"] = 80
        
        # Save cleaned parquet
        cleaned_parquet_path = parquet_path.replace(".parquet", "_cleaned.parquet")
        cleaned_df.to_parquet(cleaned_parquet_path, index=False)
        
        # Update path references
        _store_parquet_path[dataset_id + "_prev"] = parquet_path
        _store_parquet_path[dataset_id] = cleaned_parquet_path
        
        cleaned_id = dataset_id + "_cleaned"
        
        # Keep preview in memory for large datasets to prevent OOM
        if len(cleaned_df) <= 50000:
            _store[cleaned_id] = cleaned_df
        else:
            _store[cleaned_id] = cleaned_df.head(500)
            
        if goal and goal.strip():
            _store_goals[dataset_id] = goal.strip()
            
        # Preview records (up to 500 rows)
        raw_preview = raw_df.head(500)
        cleaned_preview = cleaned_df.head(500)
        diff_map = _build_diff_map(raw_preview, cleaned_preview)
        nan_map = _build_nan_map(raw_preview)
        
        missing_before = int(raw_df.isna().sum().sum())
        missing_after = int(cleaned_df.isna().sum().sum())
        
        audit_log = None
        try:
            audit_builder = AuditReportBuilder(
                raw_df=raw_df,
                cleaned_df=cleaned_df,
                cleaner_report=report,
                strategy_used=strategy_level,
                filename=_store_filename.get(dataset_id, "dataset.csv"),
                user_goal=_store_goals.get(dataset_id, goal),
                dataset_id=dataset_id,
                strategy_json=strategy_json,
            )
            audit_log = audit_builder.build()
            _audit_store[dataset_id] = audit_log
            audit_id = audit_log["audit_id"]
        except Exception as e:
            print(f"[!] Audit build error: {e}")
            audit_id = None
            audit_log = None
            
        _store_tasks[task_id]["progress"] = 90
        
        # VizEngine comparison
        try:
            viz_cmp = VizEngine(raw_df=raw_df.head(10000), cleaned_df=cleaned_df.head(10000))
            _viz_store[dataset_id] = viz_cmp.comparison()
        except Exception as _viz_err:
            print(f"[!] VizEngine comparison error (non-blocking): {_viz_err}")
            
        _store_tasks[task_id]["progress"] = 100
        _store_tasks[task_id]["status"] = "completed"
        _store_tasks[task_id]["result"] = {
            "dataset_id": dataset_id,
            "cleaned_dataset_id": cleaned_id,
            "strategy_used": strategy_level,
            "audit_id": audit_id,
            "audit_log": audit_log,
            "report": {
                **report,
                "cleaning_strategy": strategy_json.get("cleaning_strategy", {}),
            },
            "stats": {
                "rows_before": len(raw_df),
                "rows_after": len(cleaned_df),
                "missing_before": missing_before,
                "missing_after": missing_after,
                "cells_fixed": missing_before - missing_after,
            },
            "raw_data": _df_to_records(raw_preview),
            "cleaned_data": _df_to_records(cleaned_preview),
            "diff_map": diff_map,
            "nan_map": nan_map,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        _store_tasks[task_id]["status"] = "failed"
        _store_tasks[task_id]["error"] = str(e)


@app.post("/api/clean")
async def clean_dataset(
    background_tasks: BackgroundTasks,
    dataset_id: str = Form(...),
    strategy: str = Form(default="beta"),  # alpha | beta | gamma
    goal: str = Form(default=None),
    approved_actions: str = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Clean the stored dataset using a deterministic strategy or custom goal asynchronously.
    Returns a task_id to poll status.
    """
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload first.")

    strategy_level = strategy.strip().lower()
    if strategy_level not in ("alpha", "beta", "gamma"):
        raise HTTPException(status_code=422, detail="strategy must be 'alpha', 'beta', or 'gamma'.")

    # Set up previous state for Undo
    prev_id = dataset_id + "_prev"
    _store[prev_id] = _store[dataset_id].copy()
    _store_filename[prev_id] = _store_filename.get(dataset_id, "dataset.csv")
    _store_ext[prev_id] = _store_ext.get(dataset_id, ".csv")

    task_id = str(uuid.uuid4())
    _store_tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None
    }

    # Fetch file size and row/col count
    file_size = 0
    row_count = 0
    col_count = 0
    if dataset_id in _store_parquet_path:
        try:
            file_size = os.path.getsize(_store_parquet_path[dataset_id])
        except Exception:
            pass
    if dataset_id in _store:
        try:
            row_count = len(_store[dataset_id])
            col_count = len(_store[dataset_id].columns)
        except Exception:
            pass

    # Create DB record for the job
    try:
        new_job = JobRecord(
            task_id=task_id,
            user_id=current_user.id,
            task_type="cleaning",
            filename=_store_filename.get(dataset_id, "dataset.csv"),
            file_size_bytes=file_size,
            row_count=row_count,
            col_count=col_count,
            strategy=strategy_level.capitalize(),
            status="pending"
        )
        db.add(new_job)
        db.commit()
    except Exception as db_err:
        print(f"Error creating job in DB: {db_err}")

    orchestrator = KaggleStudioOrchestrator()
    background_tasks.add_task(
        orchestrator.run_remote_clean,
        task_id=task_id,
        dataset_id=dataset_id,
        strategy_level=strategy_level,
        goal=goal,
        approved_actions=approved_actions
    )

    return JSONResponse({"task_id": task_id, "status": "pending"}, status_code=202)


@app.get("/api/tasks/{task_id}/status")
def get_task_status(task_id: str, current_user: User = Depends(get_current_user)):
    """Retrieve status and progress of a background clean task."""
    if task_id not in _store_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _store_tasks[task_id]


@app.get("/api/tasks/{task_id}/report-pdf")
def get_task_pdf_report(
    task_id: str,
    locale: str = Query(default="en"),
    current_user: User = Depends(get_current_user)
):
    """Generates and downloads a PDF summary audit report for a completed clean task."""
    if task_id not in _store_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_data = _store_tasks[task_id]
    if task_data.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is in state '{task_data.get('status')}'. Report can only be generated for completed tasks."
        )
        
    is_arabic = (locale == "ar")
    
    try:
        from backend.utils.cleaning_studio_pdf_generator import CleaningStudioPDFReportGenerator
        pdf_buffer = CleaningStudioPDFReportGenerator.generate_report(
            task_id=task_id,
            task_data=task_data,
            is_arabic=is_arabic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")
        
    filename = f"SOL_Cleaning_Report_{task_id}.pdf"
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )



@app.post("/api/corrupt")
async def corrupt_dataset(
    dataset_id: str = Form(...),
    missing_pct: float = Form(default=15.0),   # 0–100
    noise_level: float = Form(default=10.0),    # 0–100
    current_user: User = Depends(get_current_user)
):
    """
    Chaos Engine — inject NaNs and format errors into the stored dataset.
    Returns a new dataset_id for the corrupted version.
    """
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _store[dataset_id].copy()
    total_cells = df.size
    rng = random.Random(42)

    null_fraction = min(max(missing_pct / 100.0, 0.0), 0.99)
    noise_fraction = min(max(noise_level / 100.0, 0.0), 0.99)

    # 1. Inject NaNs
    null_count = int(total_cells * null_fraction)
    cells = [(r, c) for r in range(len(df)) for c in range(len(df.columns))]
    chosen_null = rng.sample(cells, min(null_count, len(cells)))
    for row_idx, col_idx in chosen_null:
        df.iloc[row_idx, col_idx] = np.nan

    # 2. Inject format noise (only on object/string columns)
    noise_tokens = ["ERROR", "N/A", "??", "---", "UNKNOWN", "#VALUE!", "NULL"]
    str_cols = [c for c in df.columns if df[c].dtype == object]
    if str_cols and noise_fraction > 0:
        noise_count = int(len(df) * len(str_cols) * noise_fraction)
        str_cells = [(r, c) for r in range(len(df)) for c in str_cols]
        chosen_noise = rng.sample(str_cells, min(noise_count, len(str_cells)))
        for row_idx, col_name in chosen_noise:
            df.at[row_idx, col_name] = rng.choice(noise_tokens)

    corrupted_id = str(uuid.uuid4())
    _store[corrupted_id] = df

    nan_map = _build_nan_map(df)

    return JSONResponse({
        "corrupted_dataset_id": corrupted_id,
        "nan_map": nan_map,
        "raw_data": _df_to_records(df),
        "stats": {
            "total_cells": total_cells,
            "nulls_injected": null_count,
            "noise_injected": len(chosen_noise) if str_cols and noise_fraction > 0 else 0,
        },
    })


@app.get("/api/export/{dataset_id}")
async def export_dataset(
    dataset_id: str,
    fmt: str = Query(default="original", description="original, csv, xlsx, json, parquet, xml"),
    current_user: User = Depends(get_current_user)
):
    """Download the stored (cleaned) dataset in the requested (or original) format."""
    key = dataset_id + "_cleaned" if dataset_id + "_cleaned" in _store else dataset_id
    if key not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _store[key]
    
    if fmt == "original":
        ext = _store_ext.get(dataset_id, ".csv")
    else:
        ext = f".{fmt}"

    original_filename = _store_filename.get(dataset_id, f"sol_export_{dataset_id[:8]}")
    base_name, _ = os.path.splitext(original_filename)
    export_filename = f"{base_name}_cleaned{ext}"

    try:
        if ext in [".xlsx", ".xls"]:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            buf.seek(0)
            return StreamingResponse(
                iter([buf.read()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
        elif ext == ".json":
            json_str = df.to_json(orient="records", date_format="iso")
            return StreamingResponse(
                iter([json_str.encode("utf-8")]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
        elif ext == ".parquet":
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            return StreamingResponse(
                iter([buf.read()]),
                media_type="application/vnd.apache.parquet",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
        elif ext == ".xml":
            xml_str = df.to_xml(index=False)
            return StreamingResponse(
                iter([xml_str.encode("utf-8")]),
                media_type="application/xml",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
        elif ext == ".feather":
            buf = io.BytesIO()
            df.to_feather(buf)
            buf.seek(0)
            return StreamingResponse(
                iter([buf.read()]),
                media_type="application/vnd.apache.arrow.file",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
        elif ext == ".orc":
            buf = io.BytesIO()
            df.to_orc(buf, index=False)
            buf.seek(0)
            return StreamingResponse(
                iter([buf.read()]),
                media_type="application/x-orc",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
        else:
            # Fallback to CSV
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            export_filename = f"{base_name}_cleaned.csv"
            return StreamingResponse(
                iter([stream.getvalue().encode("utf-8")]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={export_filename}"},
            )
    except Exception as e:
        # If any format-specific export fails, fallback to standard CSV
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        export_filename = f"{base_name}_cleaned.csv"
        return StreamingResponse(
            iter([stream.getvalue().encode("utf-8")]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={export_filename}"},
        )


@app.get("/api/report/{dataset_id}")
async def get_report(dataset_id: str, current_user: User = Depends(get_current_user)):
    """Generate an AI Audit Report for a cleaned dataset."""
    cleaned_key = dataset_id + "_cleaned"
    if dataset_id not in _store or cleaned_key not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found or has not been cleaned yet.")

    raw_df = _store[dataset_id]
    cleaned_df = _store[cleaned_key]

    missing_before = int(raw_df.isna().sum().sum())
    missing_after = int(cleaned_df.isna().sum().sum())
    cells_fixed = missing_before - missing_after
    
    total_cells = raw_df.shape[0] * raw_df.shape[1]
    health_score_before = max(0, min(100, int((1 - (missing_before / total_cells)) * 100))) if total_cells > 0 else 100
    health_score_after = max(0, min(100, int((1 - (missing_after / total_cells)) * 100))) if total_cells > 0 else 100

    col_reports = []
    
    for col in raw_df.columns:
        if col not in cleaned_df.columns:
            continue
            
        b_series = raw_df[col]
        a_series = cleaned_df[col]
        
        b_missing = int(b_series.isna().sum())
        a_missing = int(a_series.isna().sum())
        fixed = b_missing - a_missing
        
        status = "Clean"
        if fixed > 0:
            status = "Fixed"
        elif a_missing > 0:
            status = "Issue Detected"
            
        action_text = f"No changes needed."
        if fixed > 0:
             action_text = f"{fixed} missing values were successfully imputed."
        elif a_missing > 0:
             action_text = f"{a_missing} missing values remain."

        stats = None
        if pd.api.types.is_numeric_dtype(b_series) and pd.api.types.is_numeric_dtype(a_series):
            stats = {
                "before": {
                   "mean": float(b_series.mean()) if not pd.isna(b_series.mean()) else None,
                   "std": float(b_series.std()) if not pd.isna(b_series.std()) else None,
                   "min": float(b_series.min()) if not pd.isna(b_series.min()) else None,
                   "max": float(b_series.max()) if not pd.isna(b_series.max()) else None,
                },
                "after": {
                   "mean": float(a_series.mean()) if not pd.isna(a_series.mean()) else None,
                   "std": float(a_series.std()) if not pd.isna(a_series.std()) else None,
                   "min": float(a_series.min()) if not pd.isna(a_series.min()) else None,
                   "max": float(a_series.max()) if not pd.isna(a_series.max()) else None,
                }
            }

        col_reports.append({
            "column": col,
            "status": status,
            "action_text": action_text,
            "missing_before": b_missing,
            "missing_after": a_missing,
            "stats": stats
        })

    return JSONResponse({
        "dataset_id": dataset_id,
        "global_metrics": {
            "health_score_increase": health_score_after - health_score_before,
            "health_score_final": health_score_after,
            "total_fixes": cells_fixed,
            "columns": len(raw_df.columns),
            "rows": len(raw_df),
            "user_goal": _store_goals.get(dataset_id),
        },
        "column_reports": col_reports
    })


@app.get("/api/datasets")
def list_datasets():
    """Return all stored dataset IDs (debug / dashboard use)."""
    datasets = []
    for key in _store.keys():
        datasets.append({
            "id": key,
            "filename": _store_filename.get(key, key)
        })
    return {
        "dataset_ids": list(_store.keys()),
        "datasets": datasets,
        "count": len(_store)
    }


class CredentialsUpdate(BaseModel):
    kaggle_username: str
    kaggle_key: str
    groq_api_key: str
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_id: Optional[str] = None

def update_env_file(updates: dict):
    env_path = os.path.join(_run_dir, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    # parse existing keys
    keys_found = {}
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if clean_line and not clean_line.startswith("#") and "=" in clean_line:
            k, v = clean_line.split("=", 1)
            keys_found[k.strip()] = i
            
    # update lines
    for key, value in updates.items():
        if key in keys_found:
            idx = keys_found[key]
            lines[idx] = f"{key}={value}\n"
        else:
            lines.append(f"{key}={value}\n")
            
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    for key, value in updates.items():
        os.environ[key] = value

@app.get("/api/settings/check-credentials")
def check_credentials(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    settings = db.query(UserSettings).filter_by(user_id=current_user.id).first()
    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    
    kaggle_user = settings.kaggle_username or ""
    kaggle_key = settings.kaggle_key or ""
    groq_key = settings.groq_api_key or ""
    elevenlabs_key = settings.elevenlabs_api_key or ""
    elevenlabs_id = settings.elevenlabs_id or ""
    
    has_kaggle_user = bool(kaggle_user.strip())
    has_kaggle_key = bool(kaggle_key.strip())
    has_groq = bool(groq_key.strip())
    has_elevenlabs_key = bool(elevenlabs_key.strip())
    has_elevenlabs_id = bool(elevenlabs_id.strip())
    
    configured = has_kaggle_user and has_kaggle_key and has_groq
    
    def mask_key(k: str) -> str:
        if not k:
            return ""
        if len(k) <= 8:
            return "*" * len(k)
        return k[:4] + "*" * (len(k) - 8) + k[-4:]
        
    return {
        "configured": configured,
        "kaggle_username": kaggle_user,
        "kaggle_key": mask_key(kaggle_key),
        "groq_api_key": mask_key(groq_key),
        "elevenlabs_api_key": mask_key(elevenlabs_key),
        "elevenlabs_id": elevenlabs_id
    }

@app.post("/api/settings/credentials")
def save_credentials(
    data: CredentialsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    settings = db.query(UserSettings).filter_by(user_id=current_user.id).first()
    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)
        
    updated_keys = []
    settings.kaggle_username = data.kaggle_username.strip()
    updated_keys.append("KAGGLE_USERNAME")
        
    if "*" not in data.kaggle_key:
        settings.kaggle_key = data.kaggle_key.strip()
        updated_keys.append("KAGGLE_KEY")
        
    if "*" not in data.groq_api_key:
        settings.groq_api_key = data.groq_api_key.strip()
        updated_keys.append("GROQ_API_KEY")

    if data.elevenlabs_api_key is not None and "*" not in data.elevenlabs_api_key:
        settings.elevenlabs_api_key = data.elevenlabs_api_key.strip()
        updated_keys.append("ELEVENLABS_API_KEY")

    if data.elevenlabs_id is not None:
        settings.elevenlabs_id = data.elevenlabs_id.strip()
        updated_keys.append("ELEVENLABS_ID")
        
    db.commit()
    return {"status": "success", "updated_keys": updated_keys}

@app.get("/api/settings/elevenlabs-usage")
def get_elevenlabs_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import urllib.request
    import json
    
    settings = db.query(UserSettings).filter_by(user_id=current_user.id).first()
    api_key = settings.elevenlabs_api_key if settings else ""
    if not api_key:
        return {"character_count": 0, "character_limit": 10000, "status": "unconfigured"}
        
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": api_key}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            character_count = res_data.get("character_count", 0)
            character_limit = res_data.get("character_limit", 10000)
            return {
                "character_count": character_count,
                "character_limit": character_limit,
                "status": "success"
            }
    except Exception as e:
        return {"character_count": 0, "character_limit": 10000, "status": f"error: {str(e)}"}

@app.get("/api/notifications")
def get_notifications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notifications = db.query(Notification).filter(
        Notification.user_id == current_user.id
    ).order_by(Notification.created_at.desc()).all()
    
    return [
        {
            "id": n.id,
            "title": n.title,
            "message": n.message,
            "type": n.type,
            "is_read": n.is_read,
            "created_at": n.created_at.isoformat() if n.created_at else None
        }
        for n in notifications
    ]

@app.post("/api/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    n = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()
    if not n:
        raise HTTPException(status_code=404, detail="Notification not found")
    n.is_read = True
    db.commit()
    return {"status": "success"}

@app.post("/api/notifications/read-all")
def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).update({"is_read": True})
    db.commit()
    return {"status": "success"}

# ─────────────────────────────────────────────────────────
#  Multi-Task Concurrent System Support (WS & REST APIs)
# ─────────────────────────────────────────────────────────
from fastapi import WebSocket, WebSocketDisconnect, status
from datetime import datetime, timezone

class ConnectionManager:
    def __init__(self):
        # Maps user_id (int) -> List[WebSocket]
        self.active_connections: dict[int, list[WebSocket]] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, user_id: int, websocket: WebSocket):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

    async def broadcast_to_user(self, user_id: int, event_type: str, data: dict):
        payload = {
            "type": event_type,
            "data": data
        }
        await self.send_personal_message(payload, user_id)

manager = ConnectionManager()

class ProjectCreateSchema(BaseModel):
    title: str
    description: Optional[str] = None

class TaskCreateSchema(BaseModel):
    name: str
    project_id: Optional[str] = None
    state_data: Optional[dict] = None

class TaskStateUpdateSchema(BaseModel):
    state_data: dict
    version_id: int

@app.websocket("/ws/tasks")
async def websocket_tasks(websocket: WebSocket, token: Optional[str] = Query(None)):
    db = SessionLocal()
    try:
        token_str = token
        if not token_str:
            token_str = websocket.cookies.get("sol_auth_token") or websocket.cookies.get("access_token")

        if not token_str:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        from backend.auth import decode_access_token
        payload = decode_access_token(token_str)
        if not payload or not payload.get("sub"):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        sub = payload.get("sub")
        if "@" in sub:
            user = db.query(User).filter(User.email == sub).first()
        else:
            user = db.query(User).filter(User.id == int(sub)).first()

        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
        user_id = user.id
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        db.close()
        return

    await manager.connect(user_id, websocket)
    try:
        while True:
            # Keep the connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)
    finally:
        db.close()

@app.get("/api/projects")
def get_projects(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    projects = db.query(Project).filter(Project.user_id == current_user.id).all()
    if not projects:
        # Auto-seed a default project
        default_proj = Project(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            title="Default Workspace Project",
            description="Automatic default workspace container"
        )
        db.add(default_proj)
        db.commit()
        db.refresh(default_proj)
        projects = [default_proj]
    return [
        {
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "created_at": p.created_at.isoformat() if p.created_at else None
        }
        for p in projects
    ]

@app.post("/api/projects")
def create_project(payload: ProjectCreateSchema, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_proj = Project(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        title=payload.title,
        description=payload.description
    )
    db.add(new_proj)
    db.commit()
    db.refresh(new_proj)
    return {
        "id": new_proj.id,
        "title": new_proj.title,
        "description": new_proj.description
    }

@app.get("/api/tasks")
def get_tasks(project_id: Optional[str] = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(Task).filter(Task.created_by == current_user.id)
    if project_id:
        query = query.filter(Task.project_id == project_id)
    tasks = query.order_by(Task.updated_at.desc()).all()
    return [
        {
            "id": t.id,
            "project_id": t.project_id,
            "name": t.name,
            "status": t.status,
            "progress_percentage": t.progress_percentage,
            "state_data": t.state_data,
            "version_id": t.version_id,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None
        }
        for t in tasks
    ]

@app.post("/api/tasks")
def create_task(payload: TaskCreateSchema, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())
    proj_id = payload.project_id
    if not proj_id:
        # Find or create default project
        default_proj = db.query(Project).filter(Project.user_id == current_user.id).first()
        if not default_proj:
            default_proj = Project(
                id=str(uuid.uuid4()),
                user_id=current_user.id,
                title="Default Workspace Project",
                description="Automatic default workspace container"
            )
            db.add(default_proj)
            db.commit()
            db.refresh(default_proj)
        proj_id = default_proj.id
    else:
        # Verify project belongs to current_user
        proj = db.query(Project).filter(Project.id == proj_id, Project.user_id == current_user.id).first()
        if not proj:
            raise HTTPException(status_code=403, detail="Not authorized to access this project")

    new_task = Task(
        id=task_id,
        project_id=proj_id,
        name=payload.name,
        state_data=payload.state_data or {},
        created_by=current_user.id,
        status="PENDING",
        progress_percentage=0,
        version_id=1
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return {
        "id": new_task.id,
        "project_id": new_task.project_id,
        "name": new_task.name,
        "status": new_task.status,
        "version_id": new_task.version_id,
        "state_data": new_task.state_data
    }

@app.put("/api/tasks/{task_id}/state")
def update_task_state(task_id: str, payload: TaskStateUpdateSchema, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.created_by == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # OCC check
    if task.version_id != payload.version_id:
        raise HTTPException(
            status_code=409, 
            detail=f"Conflict detected. Task version in database is {task.version_id}, but client provided {payload.version_id}."
        )

    task.state_data = payload.state_data
    task.version_id += 1
    db.commit()
    db.refresh(task)
    return {
        "id": task.id,
        "version_id": task.version_id,
        "state_data": task.state_data
    }

async def run_task_execution_simulation(task_id: str, user_id: int, db_session_factory):
    from backend.store import active_user_id
    active_user_id.set(user_id)
    db = db_session_factory()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            db.close()
            return

        # Create TaskRun record
        run_id = str(uuid.uuid4())
        run_record = TaskRun(
            id=run_id,
            task_id=task_id,
            user_id=user_id,
            started_at=datetime.now(timezone.utc)
        )
        db.add(run_record)
        db.commit()

        steps = [15, 35, 60, 80, 100]
        for progress in steps:
            await asyncio.sleep(2)  # Simulate 2s work per step
            
            # Re-fetch task to check if paused or cancelled
            db.refresh(task)
            if task.status == "PAUSED":
                await manager.broadcast_to_user(user_id, "task_progress", {
                    "task_id": task_id,
                    "status": "PAUSED",
                    "progress": task.progress_percentage
                })
                break

            task.progress_percentage = progress
            if progress == 100:
                task.status = "COMPLETED"
            db.commit()

            # Broadcast progress
            await manager.broadcast_to_user(user_id, "task_progress", {
                "task_id": task_id,
                "status": task.status,
                "progress": progress
            })

        if task.status == "COMPLETED":
            run_record.finished_at = datetime.now(timezone.utc)
            run_record.result_metadata = {"success": True, "details": "Processed in sandboxed directory"}
            
            # Create notification in DB
            notif = Notification(
                user_id=user_id,
                title=f"Task Completed: {task.name}",
                message=f"Your concurrent task '{task.name}' has finished executing successfully in a sandboxed directory.",
                type="success",
                is_read=False
            )
            db.add(notif)
            db.commit()

            # Broadcast notification
            await manager.broadcast_to_user(user_id, "notification", {
                "id": notif.id,
                "title": notif.title,
                "message": notif.message,
                "type": notif.type,
                "created_at": notif.created_at.isoformat() if notif.created_at else datetime.now().isoformat()
            })
            
    except Exception as e:
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "FAILED"
                db.commit()
                await manager.broadcast_to_user(user_id, "task_progress", {
                    "task_id": task_id,
                    "status": "FAILED",
                    "progress": task.progress_percentage,
                    "error": str(e)
                })
        except Exception:
            pass
    finally:
        db.close()

@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.created_by == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == "RUNNING":
        return {"status": "already_running"}

    task.status = "RUNNING"
    task.progress_percentage = 0
    db.commit()

    import asyncio
    asyncio.create_task(run_task_execution_simulation(task_id, current_user.id, SessionLocal))
    return {"status": "started", "task_id": task_id}

@app.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.created_by == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "RUNNING":
        return {"status": "not_running", "current_status": task.status}

    task.status = "PAUSED"
    db.commit()
    return {"status": "paused", "task_id": task_id}

@app.get("/api/tasks/{task_id}/runs")
def get_task_runs(task_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id, Task.created_by == current_user.id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    runs = db.query(TaskRun).filter(TaskRun.task_id == task_id).order_by(TaskRun.started_at.desc()).all()
    return [
        {
            "id": r.id,
            "task_id": r.task_id,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "error_log": r.error_log,
            "result_metadata": r.result_metadata
        }
        for r in runs
    ]

# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
# uvicorn reload trigger - 2026-06-24T01:42:00

