from fastapi import APIRouter, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.auth import get_current_user
from backend.models import User
from backend.middleware.barrier import CredentialsBarrier
from backend.store import _store, _store_filename
from backend.tools.narrator.narrator import DataNarrator
import io, json
import numpy as np
import pandas as pd

router = APIRouter(prefix="/api/narrator", tags=["Data Narrator"])
_narrator = DataNarrator()

ERROR_TOKENS = {"ERROR","error","UNKNOWN","unknown","?","-","Not Started",
                "Null","NULL","N/A","n/a","na","NA","#VALUE!","??","---","#N/A","#REF!"}

def _build_col_info(df: pd.DataFrame) -> list:
    cols = []
    for col in df.columns:
        nan_mask = df[col].isna()
        if df[col].dtype == object:
            error_mask = df[col].astype(str).str.strip().isin(ERROR_TOKENS)
            dirty = int((nan_mask | error_mask).sum())
        else:
            dirty = int(nan_mask.sum())
        pct = round(dirty / len(df) * 100, 2) if len(df) > 0 else 0
        cols.append({
            "name": col,
            "physical_type": str(df[col].dtype),
            "missing_count": dirty,
            "missing_percentage": pct,
        })
    return cols


@router.post("/pre-clean")
async def narrate_pre_clean(
    dataset_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    _barrier = Depends(CredentialsBarrier(["groq_api_key"]))
):
    """Generate Arabic pre-cleaning narrative for a stored dataset."""
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _store[dataset_id]
    filename = _store_filename.get(dataset_id, "dataset")
    metadata = {
        "rows": len(df),
        "cols": len(df.columns),
        "columns_info": _build_col_info(df),
    }

    narrative = _narrator.narrate_pre_cleaning(metadata, filename)
    return JSONResponse({"narrative": narrative, "phase": "pre"})


@router.post("/post-clean")
async def narrate_post_clean(
    dataset_id: str = Form(...),
    strategy: str = Form(default="beta"),
    stats_json: str = Form(default="{}"),
    report_json: str = Form(default="{}"),
    current_user: User = Depends(get_current_user),
    _barrier = Depends(CredentialsBarrier(["groq_api_key"]))
):
    """Generate Arabic post-cleaning narrative."""
    filename = _store_filename.get(dataset_id, "dataset")
    try:
        stats  = json.loads(stats_json)
        report = json.loads(report_json)
    except Exception:
        stats, report = {}, {}

    narrative = _narrator.narrate_post_cleaning(strategy, stats, report, filename)
    return JSONResponse({"narrative": narrative, "phase": "post"})
