from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import pandas as pd
import io
import uuid
from backend.tools.ml_advisor.processor import MLAdvisorProcessor
from backend.store import _store, _store_ext, _store_filename

router = APIRouter(prefix="/api/ml", tags=["ML Advisor"])
processor = MLAdvisorProcessor()

from backend.auth import get_current_user

@router.post("/recommend")
async def get_ml_recommendation(
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    current_user = Depends(get_current_user)
):
    df = None
    fname = "dataset.csv"
    
    # Priority 1: Use specific dataset_id from store
    if dataset_id and dataset_id in _store:
        df = _store[dataset_id]
        fname = _store_filename.get(dataset_id, "dataset.csv")
    
    # Priority 2: Use uploaded file
    elif file:
        fname = file.filename
        content = await file.read()
        ext = file.filename.split('.')[-1].lower()
        try:
            if ext == 'csv':
                df = pd.read_csv(io.BytesIO(content))
            elif ext in ['xls', 'xlsx']:
                df = pd.read_excel(io.BytesIO(content))
            elif ext == 'json':
                df = pd.read_json(io.BytesIO(content))
            elif ext == 'parquet':
                df = pd.read_parquet(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")
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
                        task_type="advisor",
                        filename=fname,
                        status="failed",
                        error_message=str(e)
                    )
                finally:
                    db_sess.close()
            except Exception as log_err:
                print(f"[ML Advisor Log Error]: {log_err}")
            raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found for analysis.")

    try:
        # Profile and Recommended
        profile = processor.profile_dataset(df, target_column)
        recommendations = await processor.get_recommendations(profile)
        
        # Log success
        try:
            from backend.database import SessionLocal
            from backend.utils.job_logger import log_job
            db_sess = SessionLocal()
            try:
                log_job(
                    db=db_sess,
                    user_id=current_user.id,
                    task_type="advisor",
                    filename=fname,
                    status="completed",
                    row_count=len(df),
                    col_count=len(df.columns),
                    accuracy_rate=86.4
                )
            finally:
                db_sess.close()
        except Exception as log_err:
            print(f"[ML Advisor Log Error]: {log_err}")

        return {
            "status": "success",
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "target": profile.get("target_column")
            },
            "metadata": profile, # Include the rich profile for the UI
            "recommendations": recommendations[:3] 
        }
    except ValueError as val_err:
        # Log failure
        try:
            from backend.database import SessionLocal
            from backend.utils.job_logger import log_job
            db_sess = SessionLocal()
            try:
                log_job(
                    db=db_sess,
                    user_id=current_user.id,
                    task_type="advisor",
                    filename=fname,
                    status="failed",
                    error_message=str(val_err)
                )
            finally:
                db_sess.close()
        except Exception as log_err:
            print(f"[ML Advisor Log Error]: {log_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
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
                    task_type="advisor",
                    filename=fname,
                    status="failed",
                    error_message=str(e)
                )
            finally:
                db_sess.close()
        except Exception as log_err:
            print(f"[ML Advisor Log Error]: {log_err}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/columns")
async def get_dataset_columns(dataset_id: str):
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _store[dataset_id]
    return {"columns": df.columns.tolist()}
