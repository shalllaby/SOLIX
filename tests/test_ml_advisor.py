import os
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app
from backend.auth import get_current_user
from backend.models import User
from backend.tools.ml_advisor.processor import MLAdvisorProcessor

# 1. Setup Auth Dependency Override
def mock_current_user():
    return User(
        id=1,
        username="test_user",
        email="test_user@example.com",
        status="active"
    )

app.dependency_overrides[get_current_user] = mock_current_user
client = TestClient(app)

def test_processor_no_api_key():
    """Verify that MLAdvisorProcessor raises ValueError if no api_key is configured."""
    processor = MLAdvisorProcessor()
    
    # Temporarily remove GROQ_API_KEY environment variable
    old_key = os.environ.get("GROQ_API_KEY")
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
        
    try:
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(processor.get_recommendations({"dummy": "profile"}))
        assert "Groq API Key is not configured" in str(exc_info.value)
    finally:
        # Restore environment variable
        if old_key is not None:
            os.environ["GROQ_API_KEY"] = old_key

@patch("backend.tools.ml_advisor.processor.AsyncGroq")
def test_processor_recommendations(mock_async_groq_class):
    """Verify that MLAdvisorProcessor correctly invokes AsyncGroq and parses JSON response."""
    mock_client = MagicMock()
    mock_async_groq_class.return_value = mock_client
    
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"recommendations": [{"model": "Decision Tree", "score": 90, "pros": "Easy to interpret", "cons": "Overfits easily", "reasoning": "Simple tabular structure"}]}'
    mock_completion.choices = [mock_choice]
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    processor = MLAdvisorProcessor(api_key="mock_key")
    profile = {"features": 10, "samples": 1000, "task": "classification"}
    
    recs = asyncio.run(processor.get_recommendations(profile))
    assert len(recs) == 1
    assert recs[0]["model"] == "Decision Tree"
    assert recs[0]["score"] == 90
    assert recs[0]["pros"] == "Easy to interpret"

def test_router_recommend_validation_error():
    """Verify that router returns 400 Bad Request when ValueError (missing key) is raised."""
    from backend.store import _store
    import pandas as pd
    
    _store["test_ds"] = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    
    with patch("backend.tools.ml_advisor.router.processor.get_recommendations", new_callable=AsyncMock) as mock_get_rec:
        mock_get_rec.side_effect = ValueError("Groq API Key is not configured. Please set it in Settings.")
        
        fd = {
            "dataset_id": "test_ds",
            "target_column": "col2"
        }
        res = client.post("/api/ml/recommend", data=fd)
        assert res.status_code == 400
        assert "Groq API Key is not configured" in res.json()["detail"]
