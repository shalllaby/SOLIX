from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.database import get_db
from backend.models import User, JobRecord, TokenUsageRecord, AuthLog, Feedback, Project, Task, Form, FormResponse
from backend.auth import get_current_user

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Helper dependency to enforce admin-only access
def require_admin(current_user: User = Depends(get_current_user)):
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Access denied. Admin privileges required.")
    return current_user

@router.get("/stats")
def get_admin_stats(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.status == "active").count()
    pending_users = db.query(User).filter(User.status == "pending_verification").count()
    total_jobs = db.query(JobRecord).count()
    total_feedback = db.query(Feedback).count()
    
    # Token usage sum
    total_tokens = db.query(func.sum(TokenUsageRecord.total_tokens)).scalar() or 0
    prompt_tokens = db.query(func.sum(TokenUsageRecord.prompt_tokens)).scalar() or 0
    completion_tokens = db.query(func.sum(TokenUsageRecord.completion_tokens)).scalar() or 0

    # Job status breakdowns
    completed_jobs = db.query(JobRecord).filter(JobRecord.status == "completed").count()
    failed_jobs = db.query(JobRecord).filter(JobRecord.status == "failed").count()
    processing_jobs = db.query(JobRecord).filter(JobRecord.status == "processing").count()
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "pending": pending_users
        },
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "processing": processing_jobs
        },
        "tokens": {
            "total": total_tokens,
            "prompt": prompt_tokens,
            "completion": completion_tokens
        },
        "feedback_count": total_feedback
    }

@router.get("/users")
def get_all_users(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    users = db.query(User).order_by(User.id.desc()).all()
    return [
        {
            "id": u.id,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "username": u.username,
            "email": u.email,
            "status": u.status,
            "is_admin": getattr(u, "is_admin", False),
            "job_title": u.job_title,
            "organization": u.organization,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "avatar_url": u.avatar_url
        }
        for u in users
    ]

@router.put("/users/{user_id}/status")
def update_user_status(
    user_id: int,
    status: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    if status not in ("active", "pending_verification"):
        raise HTTPException(status_code=400, detail="Invalid status value")
        
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    target_user.status = status
    db.commit()
    
    # Log administrative action
    log = AuthLog(
        email=current_user.email,
        action="admin_update_status",
        status="info",
        details=f"Updated status of user ID {user_id} ({target_user.email}) to {status}."
    )
    db.add(log)
    db.commit()
    
    return {"status": "success", "message": f"User status updated to {status}."}

@router.put("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    is_admin: bool = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    # Prevent self-demotion
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Admins cannot change their own roles.")
        
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    target_user.is_admin = is_admin
    db.commit()
    
    # Log administrative action
    log = AuthLog(
        email=current_user.email,
        action="admin_update_role",
        status="info",
        details=f"Updated admin role of user ID {user_id} ({target_user.email}) to {is_admin}."
    )
    db.add(log)
    db.commit()
    
    return {"status": "success", "message": f"User role updated. Admin: {is_admin}."}

@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    # Prevent self-deletion
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Admins cannot delete themselves.")
        
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Delete associated user projects, forms, tasks, notifications
    db.query(Project).filter(Project.user_id == user_id).delete()
    db.query(Form).filter(Form.user_id == user_id).delete()
    db.query(Task).filter(Task.created_by == user_id).delete()
    db.query(JobRecord).filter(JobRecord.user_id == user_id).delete()
    
    db.delete(target_user)
    db.commit()
    
    # Log administrative action
    log = AuthLog(
        email=current_user.email,
        action="admin_delete_user",
        status="warning",
        details=f"Deleted user ID {user_id} ({target_user.email}) and all their associated data."
    )
    db.add(log)
    db.commit()
    
    return {"status": "success", "message": "User and associated data permanently deleted."}

@router.get("/auth-logs")
def get_auth_logs(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    logs = db.query(AuthLog).order_by(AuthLog.id.desc()).limit(100).all()
    return [
        {
            "id": l.id,
            "ip_address": l.ip_address,
            "email": l.email,
            "user_agent": l.user_agent,
            "action": l.action,
            "status": l.status,
            "details": l.details,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None
        }
        for l in logs
    ]

@router.get("/job-records")
def get_job_records(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    jobs = db.query(JobRecord).order_by(JobRecord.created_at.desc()).limit(100).all()
    return [
        {
            "id": j.id,
            "task_id": j.task_id,
            "user_id": j.user_id,
            "task_type": j.task_type,
            "filename": j.filename,
            "file_size_bytes": j.file_size_bytes,
            "row_count": j.row_count,
            "col_count": j.col_count,
            "strategy": j.strategy,
            "status": j.status,
            "accuracy_rate": j.accuracy_rate,
            "error_message": j.error_message,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "updated_at": j.updated_at.isoformat() if j.updated_at else None
        }
        for j in jobs
    ]

@router.get("/feedbacks")
def get_feedbacks(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    feedbacks = db.query(Feedback).order_by(Feedback.created_at.desc()).limit(100).all()
    return [
        {
            "id": fb.id,
            "name": fb.name,
            "email": fb.email,
            "phone": fb.phone,
            "message": fb.message,
            "created_at": fb.created_at.isoformat() if fb.created_at else None
        }
        for fb in feedbacks
    ]

@router.get("/token-usage")
def get_token_usage(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    usages = db.query(TokenUsageRecord).order_by(TokenUsageRecord.timestamp.desc()).limit(100).all()
    return [
        {
            "id": u.id,
            "model_name": u.model_name,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
            "module_name": u.module_name,
            "timestamp": u.timestamp.isoformat() if u.timestamp else None
        }
        for u in usages
    ]
