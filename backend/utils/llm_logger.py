import json
from backend.models import TokenUsageRecord
from backend.database import SessionLocal
from datetime import datetime, timezone

def get_utc_now():
    return datetime.now(timezone.utc)

def log_token_usage(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    module_name: str = None
):
    """
    Logs raw token usage counts directly to the SQL database.
    """
    try:
        db = SessionLocal()
        prompt_val = int(prompt_tokens or 0)
        comp_val = int(completion_tokens or 0)
        total_val = prompt_val + comp_val

        from backend.store import active_user_id
        uid = active_user_id.get()

        record = TokenUsageRecord(
            model_name=model_name or "llama-3.3-70b-versatile",
            prompt_tokens=prompt_val,
            completion_tokens=comp_val,
            total_tokens=total_val,
            module_name=module_name,
            user_id=uid,
            timestamp=get_utc_now()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        db.close()
        return record
    except Exception as e:
        print(f"[llm_logger] Error saving token usage: {e}")
        return None

def log_groq_response(response, module_name: str = None):
    """
    Utility wrapper that parses a Groq API response.
    Accepts:
      - Raw dict (e.g. from urllib json response body)
      - Object response (e.g. from groq python client response)
    """
    try:
        if response is None:
            return None

        model_name = "llama-3.3-70b-versatile"
        prompt_tokens = 0
        completion_tokens = 0

        # Case 1: Dict payload (from urllib)
        if isinstance(response, dict):
            model_name = response.get("model", model_name)
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
        # Case 2: Object response (from python-groq client)
        else:
            if hasattr(response, "model"):
                model_name = response.model
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

        return log_token_usage(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            module_name=module_name
        )
    except Exception as e:
        print(f"[llm_logger] Error parsing groq response: {e}")
        return None
