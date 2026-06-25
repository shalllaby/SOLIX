from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from backend.auth import get_current_user
from backend.models import User
import uuid
import json

from backend.store import _store, _store_ext, _store_filename
from backend.tools.data_noise.engine import DataCorruptor
import pandas as pd
import numpy as np

ERROR_TOKENS = {"ERROR", "error", "UNKNOWN", "unknown", "?", "-", "Not Started", "Null", "NULL", "N/A", "n/a", "na", "NA"}

def _is_nan_or_error(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() in ERROR_TOKENS:
        return True
    return False

def _build_nan_map(df: pd.DataFrame) -> dict[str, list[int]]:
    nan_map: dict[str, list[int]] = {}
    for col in df.columns:
        bad_rows = [idx for idx, val in enumerate(df[col]) if _is_nan_or_error(val)]
        if bad_rows:
            nan_map[col] = bad_rows
    return nan_map

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", date_format="iso"))

router = APIRouter(prefix="/api/data-noise", tags=["Data Noise"])

@router.post("/process")
async def process_data_noise(
    dataset_id: str = Form(...),
    rules_json: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload first.")
        
    df = _store[dataset_id]
    
    try:
        rules = json.loads(rules_json)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid rules JSON string.")
        
    corruptor = DataCorruptor(df)
    
    try:
        for idx, rule in enumerate(rules):
            col = rule.get('column')
            if rule.get('problem_type') == "Duplications" or col == "All Columns (Row Level)":
                col = None
            prob_type = rule.get('problem_type')
            ratio = float(rule.get('ratio', 10))
            corruptor.apply_corruption(col, prob_type, ratio)
            
        corrupted_df = corruptor.get_dataframe()
    except Exception as e:
        # Log failure
        try:
            from backend.database import SessionLocal
            from backend.utils.job_logger import log_job
            db_sess = SessionLocal()
            try:
                log_job(
                    db=db_sess,
                    user_id=current_user.id,
                    task_type="noise",
                    filename=_store_filename.get(dataset_id, "dataset.csv"),
                    status="failed",
                    error_message=str(e)
                )
            finally:
                db_sess.close()
        except Exception as log_err:
            print(f"[Data Noise Log Error]: {log_err}")
        raise HTTPException(status_code=400, detail=str(e))
        
    corrupted_id = str(uuid.uuid4())
    _store[corrupted_id] = corrupted_df
    # In order for export tool to work, we save it as _cleaned as well
    _store[corrupted_id + "_cleaned"] = corrupted_df
    
    if dataset_id in _store_ext:
        _store_ext[corrupted_id] = _store_ext[dataset_id]
    if dataset_id in _store_filename:
        _store_filename[corrupted_id] = "dirty_" + _store_filename[dataset_id]

    nan_map = _build_nan_map(corrupted_df)
    
    # Log success
    try:
        from backend.database import SessionLocal
        from backend.utils.job_logger import log_job
        db_sess = SessionLocal()
        try:
            # Determine average noise ratio or rule count
            total_ratio = sum(float(r.get('ratio', 10)) for r in rules)
            avg_ratio = total_ratio / len(rules) if rules else 10.0
            log_job(
                db=db_sess,
                user_id=current_user.id,
                task_type="noise",
                filename=_store_filename.get(corrupted_id, "dirty_dataset.csv"),
                status="completed",
                row_count=len(corrupted_df),
                col_count=len(corrupted_df.columns),
                accuracy_rate=float(max(0.0, min(100.0, 100.0 - avg_ratio)))
            )
        finally:
            db_sess.close()
    except Exception as log_err:
        print(f"[Data Noise Log Error]: {log_err}")
    
    return JSONResponse({
        "dataset_id": dataset_id,
        "corrupted_dataset_id": corrupted_id,
        "raw_data": _df_to_records(corrupted_df),
        "nan_map": nan_map,
        "stats": {
            "rows_before": len(df),
            "rows_after": len(corrupted_df),
            "cols": len(corrupted_df.columns),
            "rules_applied": len(rules)
        }
    })
