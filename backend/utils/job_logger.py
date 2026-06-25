import uuid
from backend.models import JobRecord, User
from sqlalchemy.orm import Session
from datetime import datetime, timezone

def get_utc_now():
    return datetime.now(timezone.utc)

def log_job(
    db: Session,
    task_type: str,
    filename: str,
    status: str,
    user_id: int = None,
    strategy: str = None,
    accuracy_rate: float = None,
    error_message: str = None,
    row_count: int = None,
    col_count: int = None,
    file_size_bytes: int = None,
    task_id: str = None
) -> JobRecord:
    """
    Logs or updates a background job record in the database.
    If task_id is provided and matching record exists, it will update it.
    If no user_id is provided, defaults to the first user or user 1.
    """
    try:
        if not user_id:
            # Fallback: get the first user from db
            user = db.query(User).first()
            if user:
                user_id = user.id
            else:
                user_id = 1 # fallback

        if not task_id:
            task_id = str(uuid.uuid4())

        existing = db.query(JobRecord).filter(JobRecord.task_id == task_id).first()
        if existing:
            if filename is not None: existing.filename = filename
            if status is not None: existing.status = status
            if strategy is not None: existing.strategy = strategy
            if accuracy_rate is not None: existing.accuracy_rate = accuracy_rate
            if error_message is not None: existing.error_message = error_message
            if row_count is not None: existing.row_count = row_count
            if col_count is not None: existing.col_count = col_count
            if file_size_bytes is not None: existing.file_size_bytes = file_size_bytes
            existing.updated_at = get_utc_now()
            db.commit()
            db.refresh(existing)
            return existing
        else:
            job = JobRecord(
                task_id=task_id,
                user_id=user_id,
                task_type=task_type,
                filename=filename or "unknown.csv",
                file_size_bytes=file_size_bytes or 0,
                row_count=row_count,
                col_count=col_count,
                strategy=strategy or "None",
                status=status,
                accuracy_rate=accuracy_rate,
                error_message=error_message,
                created_at=get_utc_now(),
                updated_at=get_utc_now()
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            return job
    except Exception as e:
        print(f"[job_logger] Failed to log/update job record: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return None
