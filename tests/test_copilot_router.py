import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from backend.main import app
from backend.auth import get_current_user
from backend.models import User
from backend.store import (
    _store, _store_parquet_path, _store_filename, _discovery_store,
    _store_ext, _store_goals, _store_is_db, _audit_store, _viz_store
)

def mock_current_user():
    return User(
        id=1,
        username="test_user",
        email="test_user@example.com",
        status="active"
    )

app.dependency_overrides[get_current_user] = mock_current_user
client = TestClient(app)

@pytest.fixture
def populated_store(tmp_path):
    # Setup mock store data
    dataset_id = "test_uuid_123"
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    
    # Create a dummy parquet file
    temp_file = tmp_path / "test_uuid_123.parquet"
    temp_file.write_text("dummy parquet content")
    
    _store[dataset_id] = df
    _store_parquet_path[dataset_id] = str(temp_file)
    _store_filename[dataset_id] = "dummy.csv"
    _store_ext[dataset_id] = ".csv"
    _discovery_store[dataset_id] = {"some": "viz"}
    _viz_store[dataset_id] = {"cmp": "viz"}
    
    yield str(temp_file)
    
    # Clean up just in case
    _store.clear()
    _store_parquet_path.clear()
    _store_filename.clear()
    _store_ext.clear()
    _discovery_store.clear()
    _viz_store.clear()

def test_delete_datasets_endpoint(populated_store):
    parquet_file_path = populated_store
    assert os.path.exists(parquet_file_path)
    
    # Verify pre-conditions
    assert len(_store) == 1
    assert len(_store_parquet_path) == 1
    
    # Make delete request
    res = client.delete("/api/copilot/datasets")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    
    # Verify stores are empty
    assert len(_store) == 0
    assert len(_store_parquet_path) == 0
    assert len(_discovery_store) == 0
    assert len(_viz_store) == 0
    
    # Verify physical file deleted
    assert not os.path.exists(parquet_file_path)
