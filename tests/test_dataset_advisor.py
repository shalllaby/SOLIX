import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app
from backend.auth import get_current_user
from backend.models import User

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

def test_dataset_advisor_stats():
    """Verify that system stats monitoring endpoints work correctly."""
    with patch("backend.tools.dataset_advisor.router.qdrant_client") as mock_qdrant:
        mock_collection = MagicMock()
        mock_collection.collections = []
        mock_qdrant.get_collections.return_value = mock_collection

        res = client.get("/api/dataset-advisor/stats")
        assert res.status_code == 200
        data = res.json()
        assert "datasets_count" in data
        assert "qdrant_status" in data
        assert "engine_dialect" in data

@patch("backend.tools.dataset_advisor.router.retrieval_pipeline.execute_search")
def test_dataset_advisor_search(mock_execute_search):
    """Verify that the search endpoint routes inputs correctly and parses responses."""
    mock_execute_search.return_value = {
        "query_analysis": {
            "primary_kaggle_query": "arabic house pricing",
            "semantic_search_query": "arabic house pricing",
            "detected_language": "arabic",
            "task_type": "regression",
            "modality": "tabular",
            "language": "arabic",
            "min_rows": 1000,
            "max_rows": 50000,
            "expansion_queries": []
        },
        "detected_language": "arabic",
        "advisor_message": "لقد قمت بتحليل طلبك ووجدت مجموعات بيانات ممتازة:",
        "datasets": [
            {
                "id": "e6a0d421-432a-4f51-b0db-6e6ad9d20cba",
                "title": "Arabic Real Estate Dataset",
                "description": "Tabular real estate values from Riyadh",
                "url": "https://www.kaggle.com/datasets/ Riyadh-estate",
                "row_count": 12000,
                "column_count": 8,
                "license": "CC0-1.0",
                "task_type": "regression",
                "language": "arabic",
                "tags": ["real estate", "arabic", "riyadh"],
                "relevance_score": 95.5,
                "reasoning": "تحتوي هذه البيانات على مساحات وأسعار العقارات باللغة العربية وهي ممتازة للتدريب."
            }
        ]
    }

    payload = {
        "query": "I want an Arabic house pricing tabular dataset under 50000 rows",
        "session_id": "test_session"
    }
    
    res = client.post("/api/dataset-advisor/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "arabic"
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["title"] == "Arabic Real Estate Dataset"
    assert data["datasets"][0]["relevance_score"] == 95.5


