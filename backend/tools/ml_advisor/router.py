from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import pandas as pd
import io
import uuid
from backend.tools.ml_advisor.processor import MLAdvisorProcessor
from backend.store import _store, _store_ext, _store_filename

router = APIRouter(prefix="/api/ml", tags=["ML Advisor"])
processor = MLAdvisorProcessor()

@router.post("/recommend")
async def get_ml_recommendation(
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None)
):
    df = None
    
    # Priority 1: Use specific dataset_id from store
    if dataset_id and dataset_id in _store:
        df = _store[dataset_id]
    
    # Priority 2: Use uploaded file
    elif file:
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
            raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found for analysis.")

    # Profile and Recommended
    profile = processor.profile_dataset(df, target_column)
    recommendations = processor.get_recommendations(profile)
    
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

@router.get("/columns")
async def get_dataset_columns(dataset_id: str):
    if dataset_id not in _store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _store[dataset_id]
    return {"columns": df.columns.tolist()}
