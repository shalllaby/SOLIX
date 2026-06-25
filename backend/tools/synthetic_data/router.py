import sys
import builtins
import locale

# Force UTF-8 globally to prevent Windows CP1252 charmap encoding errors
locale.getpreferredencoding = lambda *args: 'utf-8'

original_open = builtins.open
def custom_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if 'b' not in mode and encoding is None:
        encoding = 'utf-8'
    return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
builtins.open = custom_open

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass

from fastapi import APIRouter, Depends, HTTPException, Form, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from backend.auth import get_current_user
from backend.models import User, JobRecord
from sqlalchemy.orm import Session
from backend.database import get_db
import uuid
import json
import time
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from backend.store import _store, _store_parquet_path, _store_tasks, _store_filename
from backend.tools.synthetic_data.engine import (
    profile_dataframe,
    generate_synthetic,
    generate_synthetic_fast,
    generate_synthetic_tvae,
    generate_synthetic_ai,
    inject_noise,
    generate_report,
    generate_privacy_report,
    generate_data_dictionary
)
from backend.tools.synthetic_data.orchestrator import KaggleSyntheticOrchestrator

router = APIRouter(prefix="/api/synthetic", tags=["Synthetic Data Studio"])

def _update_synthetic_job_db(task_id: str, status: str, error_msg: str = None, stats: dict = None):
    try:
        from backend.database import SessionLocal
        from backend.models import JobRecord
        db = SessionLocal()
        try:
            job = db.query(JobRecord).filter(JobRecord.task_id == task_id).first()
            if job:
                job.status = status
                if error_msg:
                    job.error_message = error_msg
                if stats:
                    if "total_rows" in stats:
                        job.row_count = stats["total_rows"]
                    if "total_cols" in stats:
                        job.col_count = stats["total_cols"]
                    if "fidelity_avg" in stats:
                        job.accuracy_rate = stats["fidelity_avg"]
                db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to update synthetic job in DB: {e}")

@router.get("/datasets")
def list_datasets(current_user: User = Depends(get_current_user)):
    """يرجع قائمة بجميع الملفات المرفوعة في المشروع للاختيار منها."""
    datasets = []
    for kid in list(_store_filename.keys()):
        if not kid.endswith("_cleaned") and not kid.endswith("_prev") and not kid.startswith("synthetic_"):
            datasets.append({
                "dataset_id": kid,
                "filename": _store_filename[kid]
            })
    return datasets

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """
    سماح برفع ملف بيانات مخصص للتوليد الاصطناعي وحفظه في الذاكرة.
    """
    import polars as pl
    
    filename = file.filename or "upload.csv"
    ext = os.path.splitext(filename)[1].lower()

    supported_exts = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    if ext not in supported_exts:
        raise HTTPException(status_code=400, detail=f"Format {ext} is not supported yet.")

    dataset_id = str(uuid.uuid4())
    temp_dir = "temp_snapshots"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_raw_path = os.path.join(temp_dir, f"{dataset_id}_raw{ext}")
    with open(temp_raw_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    parquet_path = os.path.join(temp_dir, f"{dataset_id}.parquet")

    try:
        if ext == ".parquet":
            shutil.copy2(temp_raw_path, parquet_path)
        elif ext == ".csv":
            try:
                pl.scan_csv(temp_raw_path, infer_schema_length=10000).sink_parquet(parquet_path)
            except Exception:
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
        raise HTTPException(status_code=422, detail=f"Failed to convert dataset: {e}")

    if os.path.exists(temp_raw_path):
        os.remove(temp_raw_path)

    # Load in pandas to ensure compatibility
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to load dataset: {e}")

    _store[dataset_id] = df if len(df) <= 50000 else df.head(500)
    _store_parquet_path[dataset_id] = parquet_path
    _store_filename[dataset_id] = filename

    return {
        "dataset_id": dataset_id,
        "filename": filename
    }

def run_local_generation_task(
    task_id: str,
    dataset_id: str,
    num_rows: int,
    model_type: str,
    epochs: int,
    null_pct: float,
    outlier_pct: float,
    user_id: int
):
    from backend.store import active_user_id
    active_user_id.set(user_id)
    try:
        parquet_path = _store_parquet_path.get(dataset_id)
        if not parquet_path or not os.path.exists(parquet_path):
            raise FileNotFoundError("الملف الأصلي غير موجود.")

        # 1. قراءة البيانات الأصلية
        df = pd.read_parquet(parquet_path)
        _store_tasks[task_id]["progress"] = 15
        
        # 2. عمل تحليل هيكلي (Profile)
        profile = profile_dataframe(df)
        _store_tasks[task_id]["progress"] = 30
        
        # 3. دالة التقدم الداخلي للتوليد
        def progress_cb(pct, msg):
            # سنقوم برفع النسبة لتتوافق مع تقدم المهمة (من 30% إلى 85%)
            mapped_pct = int(30 + pct * 55)
            _store_tasks[task_id]["progress"] = min(mapped_pct, 90)
            _store_tasks[task_id]["logs"] = _store_tasks[task_id].get("logs", []) + [msg]
            
        # 4. اختيار وتشغيل محرك التوليد
        if model_type == "basic":
            synthetic_df = generate_synthetic(df, profile, num_rows)
            _store_tasks[task_id]["progress"] = 70
        elif model_type == "gaussian_copula":
            synthetic_df = generate_synthetic_fast(df, profile, num_rows, progress_callback=progress_cb)
        elif model_type == "tvae":
            synthetic_df = generate_synthetic_tvae(df, profile, num_rows, epochs=epochs, progress_callback=progress_cb)
        elif model_type == "ctgan":
            synthetic_df = generate_synthetic_ai(df, profile, num_rows, epochs=epochs, progress_callback=progress_cb)
        elif model_type == "benchmark":
            # خوارزمية المقارنة والتقييم التلقائي
            progress_cb(0.1, "🔬 جاري تشغيل المقارنة والتقييم التلقائي...")
            
            # تشغيل basic
            progress_cb(0.2, "⚙️ تشغيل النموذج الإحصائي الأساسي...")
            df_basic = generate_synthetic(df, profile, num_rows)
            score_basic = generate_report(df, df_basic, profile)["fidelity_score"].mean()
            
            # تشغيل gaussian_copula
            progress_cb(0.4, "⚙️ تشغيل نموذج Gaussian Copula...")
            try:
                df_copula = generate_synthetic_fast(df, profile, num_rows)
                score_copula = generate_report(df, df_copula, profile)["fidelity_score"].mean()
            except Exception:
                df_copula = None
                score_copula = -1
                
            progress_cb(0.7, f"📊 النتائج: إحصائي أساسي ({score_basic:.1f}%) | Gaussian Copula ({score_copula:.1f}%)")
            
            if score_copula > score_basic and df_copula is not None:
                synthetic_df = df_copula
                progress_cb(0.9, "🏆 تم اختيار نموذج Gaussian Copula لتقديم أفضل دقة وفاء للبيانات.")
            else:
                synthetic_df = df_basic
                progress_cb(0.9, "🏆 تم اختيار النموذج الإحصائي الأساسي لتقديم أفضل أداء واستقرار.")
        else:
            raise ValueError(f"نوع نموذج توليد غير مدعوم: {model_type}")

        # 5. حقن الضوضاء
        _store_tasks[task_id]["progress"] = 85
        if null_pct > 0 or outlier_pct > 0:
            _store_tasks[task_id]["logs"] = _store_tasks[task_id].get("logs", []) + ["حقن القيم المفقودة والمتطرفة المطلوبة..."]
            synthetic_df = inject_noise(synthetic_df, profile, null_pct, outlier_pct)
            
        # 6. حساب مقاييس الجودة والخصوصية والقاموس
        _store_tasks[task_id]["progress"] = 90
        
        report_df = generate_report(df, synthetic_df, profile)
        fidelity_list = report_df.to_dict(orient="records")
        
        privacy_rows, privacy_metrics = generate_privacy_report(df, synthetic_df, profile)
        
        data_dict = generate_data_dictionary(profile, synthetic_df)
        
        # 7. حفظ البيانات محلياً
        synthetic_id = f"synthetic_{dataset_id}_{int(time.time())}"
        final_synthetic_path = parquet_path.replace(".parquet", f"_synthetic_{task_id}.parquet")
        synthetic_df.to_parquet(final_synthetic_path, index=False)
        
        # تخزين في الذاكرة
        _store[synthetic_id] = synthetic_df if len(synthetic_df) <= 50000 else synthetic_df.head(500)
        _store_parquet_path[synthetic_id] = final_synthetic_path
        
        orig_filename = _store_filename.get(dataset_id, "dataset.csv")
        base_filename = orig_filename.rsplit('.', 1)[0]
        _store_filename[synthetic_id] = f"synthetic_{base_filename}.csv"
        
        # تجهيز استعراض البيانات
        preview_df = synthetic_df.head(100).replace({np.nan: None})
        preview_records = json.loads(preview_df.to_json(orient="records"))
        
        # تحديث حالة المهمة كـ مكتملة
        _store_tasks[task_id]["progress"] = 100
        _store_tasks[task_id]["status"] = "completed"
        _store_tasks[task_id]["result"] = {
            "dataset_id": dataset_id,
            "synthetic_dataset_id": synthetic_id,
            "synthetic_filename": _store_filename[synthetic_id],
            "stats": {
                "total_rows": len(synthetic_df),
                "total_cols": len(synthetic_df.columns),
                "fidelity_avg": float(report_df["fidelity_score"].mean()) if len(report_df) > 0 else 0.0
            },
            "fidelity_report": fidelity_list,
            "privacy_report": privacy_metrics,
            "data_dict": data_dict,
            "preview_records": preview_records
        }
        _update_synthetic_job_db(task_id, "completed", stats={
            "total_rows": len(synthetic_df),
            "total_cols": len(synthetic_df.columns),
            "fidelity_avg": float(report_df["fidelity_score"].mean()) if len(report_df) > 0 else 0.0
        })
        
    except Exception as e:
        _store_tasks[task_id]["status"] = "failed"
        _store_tasks[task_id]["error"] = str(e)
        _store_tasks[task_id]["logs"] = _store_tasks[task_id].get("logs", []) + [f"خطأ أثناء التوليد المحلي: {str(e)}"]
        _update_synthetic_job_db(task_id, "failed", error_msg=str(e))

@router.post("/generate")
def generate_synthetic_data(
    background_tasks: BackgroundTasks,
    dataset_id: str = Form(...),
    num_rows: int = Form(...),
    model_type: str = Form(...),
    epochs: int = Form(30),
    null_pct: float = Form(0.0),
    outlier_pct: float = Form(0.0),
    run_env: str = Form("local"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if dataset_id not in _store_parquet_path:
        raise HTTPException(status_code=404, detail="الملف المختار غير موجود.")

    task_id = str(uuid.uuid4())
    _store_tasks[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "progress": 5,
        "logs": ["بدء إعداد عملية التوليد الاصطناعي..."],
        "model_type": model_type,
        "run_env": run_env
    }

    # Fetch original file details
    file_size = 0
    if dataset_id in _store_parquet_path:
        try:
            file_size = os.path.getsize(_store_parquet_path[dataset_id])
        except Exception:
            pass
            
    try:
        new_job = JobRecord(
            task_id=task_id,
            user_id=current_user.id,
            task_type="synthetic",
            filename=_store_filename.get(dataset_id, "dataset.csv"),
            file_size_bytes=file_size,
            strategy=model_type.capitalize(),
            status="processing"
        )
        db.add(new_job)
        db.commit()
    except Exception as db_err:
        print(f"Error creating synthetic job record: {db_err}")

    if run_env == "kaggle":
        # تشغيل سحابي عبر Kaggle
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        key = os.environ.get("KAGGLE_KEY", "").strip() or os.environ.get("KAGGLE_API_TOKEN", "").strip()
        
        if not username or not key:
            raise HTTPException(
                status_code=400, 
                detail="حساب كاجل غير مهيأ. يرجى إدخال اسم المستخدم والمفتاح في صفحة الإعدادات أولاً."
            )
            
        orchestrator = KaggleSyntheticOrchestrator(username, key)
        
        def start_remote():
            _store_tasks[task_id]["logs"].append("اتصال سحابي بـ Kaggle Cloud... جاري رفع البيانات")
            orchestrator.run_remote_generate(
                task_id=task_id,
                dataset_id=dataset_id,
                num_rows=num_rows,
                model_type=model_type,
                epochs=epochs,
                null_pct=null_pct,
                outlier_pct=outlier_pct
            )
            
        background_tasks.add_task(start_remote)
    else:
        # تشغيل محلي
        # التحقق مسبقاً إذا طلب نموذج SDV وهو غير مثبت محلياً
        if model_type in ["gaussian_copula", "tvae", "ctgan"]:
            try:
                import sdv
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="مكتبة SDV غير مثبتة محلياً لتشغيل هذا النموذج. يرجى تثبيتها أو تشغيل المهمة على سحابة Kaggle Cloud."
                )

        background_tasks.add_task(
            run_local_generation_task,
            task_id=task_id,
            dataset_id=dataset_id,
            num_rows=num_rows,
            model_type=model_type,
            epochs=epochs,
            null_pct=null_pct,
            outlier_pct=outlier_pct,
            user_id=current_user.id
        )

    return {"task_id": task_id}


def run_prompt_generation_task(
    task_id: str,
    prompt: str,
    columns: list,
    rules: list,
    num_rows: int,
    locale: str,
    user_id: int = None,
    api_key: str = None
):
    try:
        if user_id:
            from backend.store import active_user_id
            active_user_id.set(user_id)
        from backend.tools.synthetic_data.engine import generate_data_from_schema
        
        _store_tasks[task_id]["progress"] = 20
        _store_tasks[task_id]["logs"].append("⚙️ جاري قراءة هيكل الجدول المختار...")
        
        # توليد البيانات
        _store_tasks[task_id]["progress"] = 50
        _store_tasks[task_id]["logs"].append(f"🚀 البدء في توليد {num_rows:,} صف باستخدام Faker والتوزيع العشوائي...")
        
        start_time = time.time()
        synthetic_df = generate_data_from_schema(columns, rules, num_rows, locale, user_prompt=prompt, user_id=user_id, api_key=api_key)
        duration = time.time() - start_time
        
        _store_tasks[task_id]["progress"] = 85
        _store_tasks[task_id]["logs"].append(f"✅ تم توليد البيانات بنجاح في {duration:.2f} ثانية.")
        
        # حفظ الملف في الذاكرة
        synthetic_id = f"prompt_synthetic_{task_id}"
        temp_dir = "temp_snapshots"
        os.makedirs(temp_dir, exist_ok=True)
        final_synthetic_path = os.path.join(temp_dir, f"{synthetic_id}.parquet")
        synthetic_df.to_parquet(final_synthetic_path, index=False)
        
        # تخزين في الذاكرة
        _store[synthetic_id] = synthetic_df if len(synthetic_df) <= 50000 else synthetic_df.head(500)
        _store_parquet_path[synthetic_id] = final_synthetic_path
        _store_filename[synthetic_id] = "prompt_generated_dataset.csv"
        
        # تجهيز استعراض البيانات
        preview_df = synthetic_df.head(100).replace({np.nan: None})
        preview_records = json.loads(preview_df.to_json(orient="records"))
        
        # تحديث حالة المهمة كـ مكتملة
        _store_tasks[task_id]["progress"] = 100
        _store_tasks[task_id]["status"] = "completed"
        _store_tasks[task_id]["result"] = {
            "synthetic_dataset_id": synthetic_id,
            "synthetic_filename": _store_filename[synthetic_id],
            "stats": {
                "total_rows": len(synthetic_df),
                "total_cols": len(synthetic_df.columns),
            },
            "preview_records": preview_records
        }
        _update_synthetic_job_db(task_id, "completed", stats={
            "total_rows": len(synthetic_df),
            "total_cols": len(synthetic_df.columns)
        })
        
    except Exception as e:
        _store_tasks[task_id]["status"] = "failed"
        _store_tasks[task_id]["error"] = str(e)
        _store_tasks[task_id]["logs"].append(f"❌ خطأ أثناء توليد البيانات: {str(e)}")
        _update_synthetic_job_db(task_id, "failed", error_msg=str(e))


from backend.middleware.barrier import CredentialsBarrier

@router.post("/prompt/suggest-schema")
def suggest_schema(
    prompt: str = Form(...),
    num_columns: int = Form(5),
    locale: str = Form("en_US"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _barrier = Depends(CredentialsBarrier(["groq_api_key"]))
):
    from backend.tools.synthetic_data.engine import suggest_schema_from_prompt
    from backend.models import UserSettings
    try:
        settings = db.query(UserSettings).filter_by(user_id=current_user.id).first()
        groq_api_key = settings.groq_api_key if settings else None
        
        result = suggest_schema_from_prompt(prompt, num_columns, locale, api_key=groq_api_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to suggest schema: {e}")


@router.post("/prompt/generate")
def generate_from_prompt(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    columns_json: str = Form(...),
    rules_json: str = Form("[]"),
    num_rows: int = Form(...),
    locale: str = Form("en_US"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _barrier = Depends(CredentialsBarrier(["groq_api_key"]))
):
    try:
        columns = json.loads(columns_json)
        rules = json.loads(rules_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
        
    task_id = str(uuid.uuid4())
    _store_tasks[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "progress": 5,
        "logs": ["بدء توليد البيانات بناءً على الوصف النصي والهيكل المقترح..."],
        "model_type": "prompt_faker",
        "run_env": "local"
    }
    
    try:
        new_job = JobRecord(
            task_id=task_id,
            user_id=current_user.id,
            task_type="synthetic",
            filename="prompt_generated_dataset.csv",
            file_size_bytes=0,
            strategy="Prompt_faker",
            status="processing"
        )
        db.add(new_job)
        db.commit()
    except Exception as db_err:
        print(f"Error creating synthetic job record: {db_err}")
    
    from backend.models import UserSettings
    settings = db.query(UserSettings).filter_by(user_id=current_user.id).first()
    groq_api_key = settings.groq_api_key if settings else None

    background_tasks.add_task(
        run_prompt_generation_task,
        task_id=task_id,
        prompt=prompt,
        columns=columns,
        rules=rules,
        num_rows=num_rows,
        locale=locale,
        user_id=current_user.id,
        api_key=groq_api_key
    )
    
    return {"task_id": task_id}


@router.get("/tasks/{task_id}/status")
def get_task_status(task_id: str, current_user: User = Depends(get_current_user)):
    if task_id not in _store_tasks:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة.")
    return _store_tasks[task_id]

@router.get("/download/{dataset_id}")
def download_synthetic_dataset(dataset_id: str, current_user: User = Depends(get_current_user)):
    """تنزيل الملف الاصطناعي المولد كملف CSV."""
    if dataset_id not in _store_parquet_path:
        raise HTTPException(status_code=404, detail="الملف غير موجود.")
        
    parquet_path = _store_parquet_path[dataset_id]
    if not os.path.exists(parquet_path):
        raise HTTPException(status_code=404, detail="ملف البيانات غير متوفر على القرص.")
        
    df = pd.read_parquet(parquet_path)
    filename = _store_filename.get(dataset_id, "synthetic_dataset.csv")
    
    # حفظ مؤقت كـ CSV للتصدير
    temp_dir = Path("temp_snapshots")
    temp_dir.mkdir(exist_ok=True)
    temp_csv_path = temp_dir / f"{dataset_id}.csv"
    
    df.to_csv(temp_csv_path, index=False, encoding="utf-8-sig")
    
    return FileResponse(
        path=str(temp_csv_path),
        filename=filename,
        media_type="text/csv"
    )
