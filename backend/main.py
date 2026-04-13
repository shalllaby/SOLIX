import sys
import os
import uuid
import warnings
import io
import json
import random
import string

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

# ─────────────────────────────────────────────────────────
#  FastAPI
# ─────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request, Depends
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
import uvicorn

from backend.database import engine, Base, get_db
from backend.models import User
from backend.tools.ocr.router import router as ocr_router
from backend.tools.forms.router import router as form_router
from backend.tools.semantic_mapper.router import router as semantic_router
from backend.tools.ml_advisor.router import router as ml_router
from backend.tools.audit.router import router as audit_router
from backend.tools.data_noise.router import router as data_noise_router
from backend.tools.narrator.router import router as narrator_router

from backend.store import (
    _store, _store_ext, _store_filename, _store_goals, _store_is_db, _audit_store
)
from backend.tools.audit.engine import AuditReportBuilder
from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user

# Create database tables
Base.metadata.create_all(bind=engine)

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

frontend_dir = os.path.join(_run_dir, "frontend")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Mount Routers
app.include_router(ocr_router)
app.include_router(form_router)
app.include_router(semantic_router)
app.include_router(ml_router)
app.include_router(audit_router)
app.include_router(data_noise_router)
app.include_router(narrator_router)
templates = Jinja2Templates(directory=os.path.join(frontend_dir, "templates"))

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    # Automatically redirect unauthenticated HTML requests to login
    if exc.status_code == 401 and request.url.path.startswith("/app"):
        return RedirectResponse(url="/login")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# store is imported from backend.store

analyzer = MetadataAnalyzer()

from database_utils import DatabaseManager
db_manager = DatabaseManager()

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

    # Limit summaries list to prevent UI bloat
    display_summaries = summaries[:6]
    if len(summaries) > 6:
        display_summaries.append(f"... and {len(summaries) - 6} more operations")

    return {
        "remove_duplicates": remove_dups,
        "cleaning_strategy": plan,
        "summaries": display_summaries,
        "goal_injected": goal_injected
    }


# ─────────────────────────────────────────────────────────
#  Frontend Routes
# ─────────────────────────────────────────────────────────

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

@app.get("/forgot-password", response_class=HTMLResponse)
def read_forgot_password(request: Request):
    return templates.TemplateResponse("public/forgot_password.html", {"request": request})

@app.get("/reset-password", response_class=HTMLResponse)
def read_reset_password(request: Request):
    return templates.TemplateResponse("public/reset_password.html", {"request": request})

@app.get("/app/dashboard", response_class=HTMLResponse)
def read_dashboard(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/dashboard.html", {"request": request})

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

@app.get("/app/data-noise", response_class=HTMLResponse)
def read_data_noise(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/data_noise.html", {"request": request})

@app.get("/app/audit", response_class=HTMLResponse)
def read_audit_report(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/audit_report.html", {"request": request})

@app.get("/f/{form_id}", response_class=HTMLResponse)
def read_form_fill(request: Request, form_id: int):
    return templates.TemplateResponse("public/form_fill.html", {"request": request, "form_id": form_id})

@app.get("/app/connectors", response_class=HTMLResponse)
def read_db_connectors(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/db_connectors.html", {"request": request})

@app.get("/app/settings", response_class=HTMLResponse)
def read_settings(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse("app/settings.html", {"request": request})

@app.get("/solutions", response_class=HTMLResponse)
def read_solutions(request: Request):
    return templates.TemplateResponse("public/solutions.html", {"request": request})

@app.get("/docs", response_class=HTMLResponse)
def read_docs(request: Request):
    return templates.TemplateResponse("public/docs.html", {"request": request})

@app.get("/architecture", response_class=HTMLResponse)
def read_architecture(request: Request):
    return templates.TemplateResponse("public/solutions.html", {"request": request})

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

# ─────────────────────────────────────────────────────────
#  Mock Auth API Routes
# ─────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str

@app.post("/api/auth/login")
def auth_login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": str(user.id)})
    
    response = JSONResponse({
        "token": access_token, 
        "message": "Login successful", 
        "redirect": "/app/dashboard"
    })
    # Also set a cookie so the browser natively sends it when requesting HTML Pages
    response.set_cookie(key="sol_auth_token", value=access_token, httponly=False, max_age=86400 * 7)
    return response

@app.post("/api/auth/register")
def auth_register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == req.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_pwd = get_password_hash(req.password)
    new_user = User(
        first_name=req.first_name,
        last_name=req.last_name,
        email=req.email,
        hashed_password=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": str(new_user.id)})
    response = JSONResponse({
        "token": access_token, 
        "message": "Account created successfully"
    })
    response.set_cookie(key="sol_auth_token", value=access_token, httponly=False, max_age=86400 * 7)
    return response

@app.get("/api/auth/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "email": current_user.email
    }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """
    Accept CSV or XLSX/XLS, analyse it, store the raw DataFrame, return a data profile.
    """
    content = await file.read()
    filename = file.filename or "upload.csv"
    ext = os.path.splitext(filename)[1].lower()

    supported_exts = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    if ext not in supported_exts:
        raise HTTPException(status_code=400, detail=f"Format {ext} is not supported yet.")

    try:
        metadata = analyzer.analyze_file(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Load DataFrame into store
    try:
        from data_layer.loaders.universal_loader import DataLoaderFactory
        df = DataLoaderFactory.load_data(io.BytesIO(content), file_name=filename)
    except Exception:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content))
        elif ext == ".json":
            df = pd.read_json(io.BytesIO(content))
        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))

    dataset_id = str(uuid.uuid4())
    _store[dataset_id] = df
    _, ext = os.path.splitext(filename)
    _store_ext[dataset_id] = ext.lower()
    _store_filename[dataset_id] = filename

    # Build nan_map for the BEFORE grid (uploaded data = raw)
    nan_map = _build_nan_map(df)

    return JSONResponse({
        "dataset_id": dataset_id,
        "metadata": metadata,
        "nan_map": nan_map,
        "raw_data": _df_to_records(df),
    })


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

@app.post("/api/clean")
async def clean_dataset(
    dataset_id: str = Form(...),
    strategy: str = Form(default="beta"),  # alpha | beta | gamma
    goal: str = Form(default=None),
    current_user: User = Depends(get_current_user)
):
    """
    Clean the stored dataset using a deterministic strategy or custom goal.
    Returns cleaned records + diff_map for the comparison grid.
    """
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload first.")

    raw_df = _store[dataset_id]
    strategy_level = strategy.strip().lower()
    if strategy_level not in ("alpha", "beta", "gamma"):
        raise HTTPException(status_code=422, detail="strategy must be 'alpha', 'beta', or 'gamma'.")

    # Build metadata for strategy planner
    buffer = io.BytesIO()
    raw_df.to_csv(buffer, index=False)
    buffer.seek(0)
    try:
        metadata = analyzer.analyze_file(buffer.read(), "data.csv")
    except Exception:
        metadata = {"columns_info": []}

    strategy_json = _build_strategy(strategy_level, metadata, goal)

    # Execute cleaning
    cleaner = SmartDataCleaner(raw_df)
    result = cleaner.execute_strategy(strategy_json)
    cleaned_df, report = result if isinstance(result, tuple) else (result, {"actions": []})

    # Build diff_map and nan_map
    diff_map = _build_diff_map(raw_df, cleaned_df)
    nan_map = _build_nan_map(raw_df)

    # Store cleaned df under a new id, but also keyed for export
    cleaned_id = dataset_id + "_cleaned"
    _store[cleaned_id] = cleaned_df

    if goal and goal.strip():
        _store_goals[dataset_id] = goal.strip()

    missing_before = int(raw_df.isna().sum().sum())
    missing_after = int(cleaned_df.isna().sum().sum())

    # ── Build & persist Audit Log ─────────────────────────────────────────
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
        # Never let audit failure break the clean response
        print(f"[!] Audit build error: {e}")
        audit_id = None
    # ─────────────────────────────────────────────────────────────────────

    return JSONResponse({
        "dataset_id": dataset_id,
        "cleaned_dataset_id": cleaned_id,
        "strategy_used": strategy_level,
        "audit_id": audit_id,
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
        "raw_data": _df_to_records(raw_df),
        "cleaned_data": _df_to_records(cleaned_df),
        "diff_map": diff_map,
        "nan_map": nan_map,
    })


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
    return {"dataset_ids": list(_store.keys()), "count": len(_store)}


# ─────────────────────────────────────────────────────────
#  Live Database Connectors
# ─────────────────────────────────────────────────────────
class ConnectRequest(BaseModel):
    db_type: str
    host: str
    port: str
    user: str
    password: str
    db_name: str

@app.post("/api/connect")
def connect_db(req: ConnectRequest):
    conn_id = str(uuid.uuid4())
    db_manager.connect(conn_id, req.db_type, req.host, req.port, req.user, req.password, req.db_name)
    return {"status": "Success", "connection_id": conn_id}

@app.get("/api/db/tables")
def get_db_tables(connection_id: str):
    return {"tables": db_manager.get_tables(connection_id)}

class ImportRequest(BaseModel):
    connection_id: str
    table_name: str

@app.post("/api/db/import")
def import_db_table(req: ImportRequest):
    df = db_manager.import_table(req.connection_id, req.table_name)
    dataset_id = str(uuid.uuid4())
    _store[dataset_id] = df
    _store_ext[dataset_id] = ".csv"
    _store_filename[dataset_id] = f"{req.table_name}.csv"
    _store_is_db[dataset_id] = True
    
    # Analyze metadata
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    try:
        metadata = analyzer.analyze_file(buffer.read(), f"{req.table_name}.csv")
    except Exception:
        metadata = {"columns_info": []}

    nan_map = _build_nan_map(df)
    return {
        "dataset_id": dataset_id,
        "metadata": metadata,
        "nan_map": nan_map,
        "raw_data": _df_to_records(df),
        "is_db": True,
        "table_name": req.table_name,
        "connection_id": req.connection_id
    }

class SyncRequest(BaseModel):
    dataset_id: str
    connection_id: str
    table_name: str
    mode: str = "replace"

@app.post("/api/db/sync")
def sync_db_table(req: SyncRequest):
    cleaned_key = req.dataset_id + "_cleaned"
    if cleaned_key not in _store:
        raise HTTPException(status_code=404, detail="Cleaned dataset not found.")
    df = _store[cleaned_key]
    db_manager.sync_table(req.connection_id, req.table_name, df, if_exists=req.mode)
    return {"status": "Success"}

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
