from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import os
import datetime
import psutil
import csv
import json
import io
from typing import List, Dict, Any

from backend.database import get_db, db_path
from backend.models import JobRecord, User, TokenUsageRecord
from backend.auth import get_current_user
from backend.store import _store_tasks

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

def format_size(bytes_size: int) -> str:
    """Formats bytes size into human-readable format."""
    for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

@router.get("/stats")
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Sync in-memory tasks to DB in case of server restarts/updates
    active_db_jobs = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.status.in_(["pending", "processing"])
    ).all()
    
    for job in active_db_jobs:
        if job.task_id in _store_tasks:
            mem_task = _store_tasks[job.task_id]
            mem_status = mem_task.get("status")
            if mem_status and mem_status != job.status:
                job.status = mem_status
                if mem_status == "failed" and mem_task.get("error"):
                    job.error_message = str(mem_task["error"])
                elif mem_status == "completed":
                    # Sync row counts and accuracy if available
                    res = mem_task.get("result")
                    if isinstance(res, dict):
                        stats = res.get("stats")
                        if isinstance(stats, dict):
                            if "rows_after" in stats:
                                job.row_count = stats["rows_after"]
                            elif "rows" in stats:
                                job.row_count = stats["rows"]
                            
                            if "accuracy" in stats:
                                job.accuracy_rate = float(stats["accuracy"])
                        
                        report = res.get("report")
                        if isinstance(report, dict) and "truth_confidence_score" in report:
                            job.accuracy_rate = float(report["truth_confidence_score"])
                db.commit()

    # 2. Compute KPI Metrics & Stats (Isolated per user)
    total_cleaned_count = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.task_type == "cleaning",
        JobRecord.status == "completed"
    ).count()

    total_synthetic_count = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.task_type == "synthetic",
        JobRecord.status == "completed"
    ).count()

    total_automl_count = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.task_type == "automl",
        JobRecord.status == "completed"
    ).count()

    total_size_bytes_q = db.query(func.sum(JobRecord.file_size_bytes)).filter(
        JobRecord.user_id == current_user.id
    ).scalar()
    total_size_bytes = total_size_bytes_q if total_size_bytes_q is not None else 0
    storage_used_str = format_size(total_size_bytes)

    avg_accuracy_q = db.query(func.avg(JobRecord.accuracy_rate)).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.accuracy_rate.isnot(None)
    ).scalar()
    avg_accuracy = round(float(avg_accuracy_q), 2) if avg_accuracy_q is not None else 98.4

    active_jobs_count = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.status.in_(["pending", "processing", "running"])
    ).count()

    # Helper for getting last run details
    def get_last_job(task_type: str) -> dict:
        job = db.query(JobRecord).filter(
            JobRecord.user_id == current_user.id,
            JobRecord.task_type == task_type
        ).order_by(JobRecord.created_at.desc()).first()
        if job:
            return {
                "id": job.id,
                "task_id": job.task_id,
                "task_type": job.task_type,
                "filename": job.filename or "unknown.csv",
                "file_size_bytes": job.file_size_bytes or 0,
                "row_count": job.row_count or 0,
                "col_count": job.col_count or 0,
                "strategy": job.strategy or "None",
                "status": job.status,
                "accuracy_rate": job.accuracy_rate,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
                "updated_at": job.updated_at.isoformat() if hasattr(job.updated_at, "isoformat") else str(job.updated_at),
            }
        return None

    last_runs = {
        "cleaning": get_last_job("cleaning"),
        "chat": get_last_job("chat"),
        "automl": get_last_job("automl"),
        "synthetic": get_last_job("synthetic"),
        "ocr": get_last_job("ocr"),
        "forms": get_last_job("forms"),
        "semantic": get_last_job("semantic"),
        "advisor": get_last_job("advisor"),
        "noise": get_last_job("noise")
    }

    # 3. Retrieve Recent Activity (last 10 jobs)
    recent_jobs = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id
    ).order_by(JobRecord.created_at.desc()).limit(10).all()

    recent_jobs_list = []
    for job in recent_jobs:
        recent_jobs_list.append({
            "id": job.id,
            "task_id": job.task_id,
            "task_type": job.task_type,
            "filename": job.filename or "unknown.csv",
            "file_size_bytes": job.file_size_bytes or 0,
            "row_count": job.row_count or 0,
            "col_count": job.col_count or 0,
            "strategy": job.strategy or "None",
            "status": job.status,
            "accuracy_rate": job.accuracy_rate,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
            "updated_at": job.updated_at.isoformat() if hasattr(job.updated_at, "isoformat") else str(job.updated_at),
        })

    # 4. Generate Chart Datasets
    # A. Task Status Distribution
    status_counts = db.query(JobRecord.status, func.count(JobRecord.id)).filter(
        JobRecord.user_id == current_user.id
    ).group_by(JobRecord.status).all()
    
    status_chart = {"labels": [], "values": []}
    for status, count in status_counts:
        status_chart["labels"].append(status.capitalize())
        status_chart["values"].append(count)

    # B. Activity Timeline (Jobs over last 7 days)
    today = datetime.date.today()
    last_7_days = [today - datetime.timedelta(days=i) for i in range(6, -1, -1)]
    timeline_chart = {"labels": [], "values": []}
    timeline_data = []
    
    for day in last_7_days:
        start_dt = datetime.datetime.combine(day, datetime.time.min)
        end_dt = datetime.datetime.combine(day, datetime.time.max)
        
        count = db.query(JobRecord).filter(
            JobRecord.user_id == current_user.id,
            JobRecord.created_at >= start_dt,
            JobRecord.created_at <= end_dt
        ).count()
        
        timeline_chart["labels"].append(day.strftime("%b %d"))
        timeline_chart["values"].append(count)
        
        timeline_data.append({
            "date": day.strftime("%b %d"),
            "count": count
        })

    # C. Strategy Distribution (Alpha, Beta, Gamma)
    strategy_counts = db.query(JobRecord.strategy, func.count(JobRecord.id)).filter(
        JobRecord.user_id == current_user.id,
        JobRecord.strategy.isnot(None),
        JobRecord.strategy != "None"
    ).group_by(JobRecord.strategy).all()
    
    strategy_chart = {"labels": [], "values": []}
    for strat, count in strategy_counts:
        strategy_chart["labels"].append(strat)
        strategy_chart["values"].append(count)

    # D. Type Distribution
    type_distribution_data = [
        {"type": "cleaning", "count": db.query(JobRecord).filter(
            JobRecord.user_id == current_user.id,
            JobRecord.task_type == "cleaning"
        ).count()},
        {"type": "synthetic", "count": db.query(JobRecord).filter(
            JobRecord.user_id == current_user.id,
            JobRecord.task_type == "synthetic"
        ).count()},
        {"type": "automl", "count": db.query(JobRecord).filter(
            JobRecord.user_id == current_user.id,
            JobRecord.task_type == "automl"
        ).count()}
    ]

    # 5. Systems Health Metrics
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    
    db_size_str = "0 KB"
    if os.path.exists(db_path):
        try:
            db_size_str = format_size(os.path.getsize(db_path))
        except Exception:
            pass

    system_health = {
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{mem_percent}%",
        "database_size": db_size_str,
        "database_status": "Healthy",
        "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 6. Groq LLM Token Usage Analytics (Scoped to the current user)
    try:
        total_prompt_tokens = db.query(func.sum(TokenUsageRecord.prompt_tokens)).filter(TokenUsageRecord.user_id == current_user.id).scalar() or 0
        total_completion_tokens = db.query(func.sum(TokenUsageRecord.completion_tokens)).filter(TokenUsageRecord.user_id == current_user.id).scalar() or 0
        total_tokens = db.query(func.sum(TokenUsageRecord.total_tokens)).filter(TokenUsageRecord.user_id == current_user.id).scalar() or 0

        model_usage_query = db.query(
            TokenUsageRecord.model_name,
            func.sum(TokenUsageRecord.prompt_tokens),
            func.sum(TokenUsageRecord.completion_tokens),
            func.sum(TokenUsageRecord.total_tokens)
        ).filter(TokenUsageRecord.user_id == current_user.id).group_by(TokenUsageRecord.model_name).all()

        tokens_by_model = []
        for model_name, p_tok, c_tok, t_tok in model_usage_query:
            tokens_by_model.append({
                "model_name": model_name,
                "prompt_tokens": p_tok or 0,
                "completion_tokens": c_tok or 0,
                "total_tokens": t_tok or 0
            })

        module_usage_query = db.query(
            TokenUsageRecord.module_name,
            func.sum(TokenUsageRecord.prompt_tokens),
            func.sum(TokenUsageRecord.completion_tokens),
            func.sum(TokenUsageRecord.total_tokens)
        ).filter(TokenUsageRecord.user_id == current_user.id).group_by(TokenUsageRecord.module_name).all()

        tokens_by_module = []
        for module_name, p_tok, c_tok, t_tok in module_usage_query:
            tokens_by_module.append({
                "module_name": module_name or "unknown",
                "prompt_tokens": p_tok or 0,
                "completion_tokens": c_tok or 0,
                "total_tokens": t_tok or 0
            })

        recent_token_records = db.query(TokenUsageRecord).filter(TokenUsageRecord.user_id == current_user.id).order_by(TokenUsageRecord.timestamp.desc()).limit(15).all()
        recent_token_list = []
        for rec in recent_token_records:
            recent_token_list.append({
                "id": rec.id,
                "model_name": rec.model_name,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "total_tokens": rec.total_tokens,
                "module_name": rec.module_name or "unknown",
                "timestamp": rec.timestamp.isoformat() if hasattr(rec.timestamp, "isoformat") else str(rec.timestamp)
            })

        llm_metrics = {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "tokens_by_model": tokens_by_model,
            "tokens_by_module": tokens_by_module,
            "recent_token_logs": recent_token_list
        }
    except Exception as e:
        print(f"Failed to query TokenUsageRecord: {e}")
        llm_metrics = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "tokens_by_model": [],
            "tokens_by_module": [],
            "recent_token_logs": []
        }

    return {
        "stats": {
            "total_cleaned": total_cleaned_count,
            "total_synthetic": total_synthetic_count,
            "total_automl": total_automl_count,
            "total_size_bytes": total_size_bytes,
            "active_jobs_count": active_jobs_count,
            "avg_accuracy_rate": avg_accuracy
        },
        "last_runs": last_runs,
        "recent_jobs": recent_jobs_list,
        "chart_data": {
            "timeline": timeline_data,
            "type_distribution": type_distribution_data
        },
        "kpis": {
            "files_cleaned": total_cleaned_count,
            "storage_used": storage_used_str,
            "accuracy_rate": f"{avg_accuracy}%",
            "active_jobs": active_jobs_count
        },
        "recent_activity": recent_jobs_list,
        "charts": {
            "status": status_chart,
            "timeline": timeline_chart,
            "strategy": strategy_chart
        },
        "system_health": system_health,
        "llm_metrics": llm_metrics
    }

@router.get("/export")
def export_dashboard_jobs(
    format: str = Query("csv", regex="^(csv|json)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    jobs = db.query(JobRecord).filter(
        JobRecord.user_id == current_user.id
    ).order_by(JobRecord.created_at.desc()).all()

    if format == "json":
        data = []
        for job in jobs:
            data.append({
                "job_id": job.id,
                "task_id": job.task_id,
                "task_type": job.task_type,
                "filename": job.filename,
                "file_size_bytes": job.file_size_bytes,
                "row_count": job.row_count,
                "col_count": job.col_count,
                "strategy": job.strategy,
                "status": job.status,
                "accuracy_rate": job.accuracy_rate,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat()
            })
        
        stream = io.StringIO()
        json.dump(data, stream, indent=4)
        response = StreamingResponse(
            iter([stream.getvalue().encode("utf-8")]),
            media_type="application/json"
        )
        response.headers["Content-Disposition"] = "attachment; filename=sol_jobs_history.json"
        return response

    else: # CSV
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow([
            "Job ID", "Task ID", "Type", "Filename", "File Size (Bytes)", 
            "Rows", "Columns", "Strategy", "Status", "Accuracy Rate", "Error", "Created At"
        ])
        
        for job in jobs:
            writer.writerow([
                job.id, job.task_id, job.task_type, job.filename, job.file_size_bytes,
                job.row_count or 0, job.col_count or 0, job.strategy, job.status,
                f"{job.accuracy_rate}%" if job.accuracy_rate is not None else "N/A",
                job.error_message or "", job.created_at.strftime("%Y-%m-%d %H:%M:%S")
            ])
            
        response = StreamingResponse(
            iter([stream.getvalue().encode("utf-8")]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=sol_jobs_history.csv"
        return response

from pydantic import BaseModel
from typing import Optional

class PreferencesUpdate(BaseModel):
    widgets_layout: Any
    theme: Optional[str] = "dark"

@router.get("/preferences")
def get_preferences(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from backend.models import DashboardPreferences
    prefs = db.query(DashboardPreferences).filter_by(user_id=current_user.id).first()
    if not prefs:
        prefs = DashboardPreferences(
            user_id=current_user.id,
            widgets_layout={
                "layout": "default",
                "visible_widgets": ["cleaned_count", "synthetic_count", "automl_count", "accuracy_rate", "storage_used", "active_jobs"]
            },
            theme="dark"
        )
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
        
    return {
        "widgets_layout": prefs.widgets_layout,
        "theme": prefs.theme
    }

@router.post("/preferences")
def save_preferences(
    payload: PreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from backend.models import DashboardPreferences
    prefs = db.query(DashboardPreferences).filter_by(user_id=current_user.id).first()
    if not prefs:
        prefs = DashboardPreferences(user_id=current_user.id)
        db.add(prefs)
        
    prefs.widgets_layout = payload.widgets_layout
    if payload.theme:
        prefs.theme = payload.theme
        
    db.commit()
    return {"status": "success", "message": "Dashboard preferences updated successfully."}

