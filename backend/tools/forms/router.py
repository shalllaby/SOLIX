from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import csv
import io
from fastapi.responses import StreamingResponse

from backend.database import get_db
from backend.models import Form, FormResponse, User
from backend.auth import get_current_user, get_optional_current_user

router = APIRouter(prefix="/api/forms", tags=["forms"])

@router.post("")
def create_form(
    title: str = Body(...),
    description: str = Body(...),
    questions: List[Dict[str, Any]] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Only logged in users can create forms
):
    new_form = Form(
        title=title,
        description=description,
        questions=questions,
        user_id=current_user.id
    )
    db.add(new_form)
    db.commit()
    db.refresh(new_form)

    # Log the form creation job
    try:
        from backend.utils.job_logger import log_job
        log_job(
            db=db,
            user_id=current_user.id,
            task_type="forms",
            filename=f"Form: {title}",
            status="completed",
            accuracy_rate=100.0
        )
    except Exception as log_err:
        print(f"[Forms Log Error]: {log_err}")

    return {"status": "success", "form_id": new_form.id, "message": "Form created successfully!"}

@router.get("")
def list_forms(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Get answers count for each form too
    forms = db.query(Form).filter(Form.user_id == current_user.id).order_by(Form.id.desc()).all()
    results = []
    for f in forms:
        count = db.query(FormResponse).filter(FormResponse.form_id == f.id).count()
        results.append({
            "id": f.id,
            "title": f.title,
            "description": f.description,
            "created_at": f.created_at,
            "responses_count": count
        })
    return {"forms": results}

@router.get("/{form_id}")
def get_form(form_id: int, db: Session = Depends(get_db)): # No auth required for getting the form UI itself
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    return {
        "id": form.id,
        "title": form.title,
        "description": form.description,
        "questions": form.questions,
        "created_at": form.created_at
    }

@router.post("/{form_id}/responses")
def submit_response(
    form_id: int,
    answers: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    
    new_response = FormResponse(
        form_id=form_id,
        answers=answers,
        user_id=current_user.id if current_user else None
    )
    db.add(new_response)
    db.commit()

    # Log the response submission job
    try:
        from backend.utils.job_logger import log_job
        log_job(
            db=db,
            user_id=form.user_id,
            task_type="forms",
            filename=f"Submission: {form.title}",
            status="completed",
            accuracy_rate=88.0
        )
    except Exception as log_err:
        print(f"[Forms Log Error]: {log_err}")

    return {"status": "success", "message": "Response recorded successfully."}

@router.get("/{form_id}/responses")
def get_form_responses(form_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    if form.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this form")
        
    responses = db.query(FormResponse).filter(FormResponse.form_id == form_id).order_by(FormResponse.id.desc()).all()
    return {
        "responses": [{"id": r.id, "answers": r.answers, "timestamp": r.timestamp} for r in responses]
    }

@router.get("/{form_id}/export")
def export_form_responses(form_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    if form.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this form")
        
    responses = db.query(FormResponse).filter(FormResponse.form_id == form_id).all()
    
    # Collect all unique columns from the JSON answers
    all_keys = set()
    for r in responses:
        if isinstance(r.answers, dict):
            all_keys.update(r.answers.keys())
            
    header = ["response_id", "timestamp"] + list(all_keys)
    
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader()
    
    for r in responses:
        row = {"response_id": r.id, "timestamp": r.timestamp}
        if isinstance(r.answers, dict):
            row.update(r.answers)
        writer.writerow(row)
        
    response = StreamingResponse(
        iter([stream.getvalue().encode("utf-8")]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=form_{form_id}_responses.csv"
    return response

@router.delete("/{form_id}")
def delete_form(form_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    if form.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this form")
    
    # Delete associated responses first
    db.query(FormResponse).filter(FormResponse.form_id == form_id).delete()
    
    # Delete the form
    db.delete(form)
    db.commit()
    return {"status": "success", "message": "Form and associated data permanently deleted."}

@router.put("/{form_id}")
def update_form(
    form_id: int,
    title: str = Body(...),
    description: str = Body(...),
    questions: List[Dict[str, Any]] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    form = db.query(Form).filter(Form.id == form_id).first()
    if not form:
        raise HTTPException(status_code=404, detail="Form not found")
    if form.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this form")
    
    form.title = title
    form.description = description
    form.questions = questions
    db.commit()
    return {"status": "success", "message": "Form updated successfully!"}
