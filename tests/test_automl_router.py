import os
import io
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
from backend.store import _store, _store_ext, _store_filename

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

@pytest.fixture
def clean_store():
    _store.clear()
    _store_ext.clear()
    _store_filename.clear()
    yield
    _store.clear()
    _store_ext.clear()
    _store_filename.clear()

def test_list_datasets_empty(clean_store):
    res = client.get("/api/automl/datasets")
    assert res.status_code == 200
    assert "datasets" in res.json()
    assert len(res.json()["datasets"]) == 0

def test_direct_upload_and_profile(clean_store):
    # Create sample CSV in-memory
    csv_content = "col_a,col_b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0"
    file = ("sample.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    
    # 1. Post to upload-direct
    res = client.post("/api/automl/upload-direct", files={"file": file})
    assert res.status_code == 200
    data = res.json()
    assert "dataset_id" in data
    assert data["filename"] == "sample.csv"
    assert data["shape"]["rows"] == 5
    assert data["shape"]["cols"] == 3
    assert "target" in data["column_details"]
    assert len(data["ranked_targets"]) > 0

    dataset_id = data["dataset_id"]

    # 2. Check list_datasets contains our dataset
    res_list = client.get("/api/automl/datasets")
    assert res_list.status_code == 200
    datasets = res_list.json()["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["id"] == dataset_id
    assert datasets[0]["filename"] == "sample.csv"

    # 3. Post to profile of existing dataset
    res_prof = client.post("/api/automl/profile", json={"dataset_id": dataset_id})
    assert res_prof.status_code == 200
    prof_data = res_prof.json()
    assert prof_data["dataset_id"] == dataset_id
    assert prof_data["shape"]["rows"] == 5

def test_triage(clean_store):
    _store["test_id"] = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    _store_filename["test_id"] = "test.csv"
    
    res = client.post("/api/automl/triage", json={
        "dataset_id": "test_id",
        "target_col": "target",
        "task_type": "binary"
    })
    assert res.status_code == 200
    triage = res.json()
    assert "approved_models" in triage
    assert isinstance(triage["approved_models"], list)

@patch("backend.tools.automl.router.AutoMLTrainingEngine")
def test_train_and_export_flow(mock_engine_cls, clean_store):
    # Ensure Kaggle credentials are NOT in the environment so it executes the local training path
    env_mock = os.environ.copy()
    env_mock.pop("KAGGLE_USERNAME", None)
    env_mock.pop("KAGGLE_KEY", None)
    env_mock.pop("KAGGLE_API_TOKEN", None)
    with patch.dict(os.environ, env_mock, clear=True):
        # Mock behavior of the training engine to bypass heavy calculations and models
        mock_engine = MagicMock()
        mock_engine.model_status = "Trained & Stable"
        mock_engine.select_smart_models.return_value = ["random_forest"]
        
        leaderboard_mock = [{
            "model_name": "Random Forest Classifier",
            "fit_time": 0.05,
            "train_metrics": {"accuracy": 1.0, "f1": 1.0},
            "val_metrics": {"accuracy": 1.0, "f1": 1.0},
            "cv_mean": 0.95,
            "cv_std": 0.02,
            "generalization_gap": 0.0,
            "composite_score": 0.98,
            "status_indicator": "Stable",
            "labels": ["Stable"]
        }]
        
        dummy_model = MagicMock()
        dummy_preprocessor = MagicMock()
        dummy_encoder = MagicMock()
        
        mock_engine.train_baselines.return_value = (leaderboard_mock, {"Random Forest Classifier": dummy_model})
        mock_engine.extract_feature_importance.return_value = [
            {"feature": "col_a", "importance": 0.7},
            {"feature": "col_b", "importance": 0.3}
        ]
        
        mock_engine_cls.return_value = mock_engine
        
        # Put sample dataset in store with 15 rows to satisfy split stratification
        _store["test_id"] = pd.DataFrame({
            "col_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "col_b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        _store_filename["test_id"] = "test.csv"
        _store_ext["test_id"] = ".csv"

        # Train
        train_payload = {
            "dataset_id": "test_id",
            "target_col": "target",
            "task_type": "binary",
            "enable_deep_optimize": False
        }
        
        res = client.post("/api/automl/train", json=train_payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        
        import json
        events = []
        for line in res.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                
        assert len(events) >= 2
        assert events[0]["event"] == "kaggle_url"
        
        result_event = events[-1]
        assert result_event["event"] == "result"
        train_data = result_event["data"]
        assert train_data["status"] == "success"
        assert train_data["best_model_name"] == "Random Forest Classifier"
        assert len(train_data["leaderboard"]) == 1
        assert len(train_data["feature_importance"]) == 2

        # Export ZIP and check file stream response
        with patch("backend.tools.automl.router.AutoMLArtifactExporter.serialize_to_zip") as mock_zip:
            zip_bytes = io.BytesIO(b"fake_zip_bytes")
            mock_zip.return_value = zip_bytes
            
            res_exp = client.post("/api/automl/export", json={"dataset_id": "test_id", "is_arabic": False})
            assert res_exp.status_code == 200
            assert res_exp.headers["content-type"] == "application/zip"
            assert b"fake_zip_bytes" in res_exp.content


@patch("backend.tools.automl.router.AutoMLTrainingEngine")
@patch("core.automl.kaggle_client.KaggleWorkflowManager")
def test_kaggle_training_flow_event_ordering(mock_kaggle_mgr_cls, mock_engine_cls, clean_store):
    # Setup Kaggle credentials in env mock
    with patch.dict(os.environ, {"KAGGLE_USERNAME": "mock_user", "KAGGLE_KEY": "mock_key"}):
        # Mock KaggleWorkflowManager instance
        mock_kaggle_mgr = MagicMock()
        mock_kaggle_mgr_cls.return_value = mock_kaggle_mgr
        
        # Mock async generators
        async def mock_upload(*args, **kwargs):
            yield {"step": "Kaggle Upload", "desc": "Uploading dataset splits...", "percent": 30}
            yield {"event": "completed", "dataset_ref": "mock_user/dataset-automl"}
            
        async def mock_trigger(*args, **kwargs):
            yield {"step": "Kaggle Kernel", "desc": "Pushing kernel code...", "percent": 49}
            yield {"event": "kaggle_url", "url": "https://www.kaggle.com/code/mock_user/kernel-automl"}
            yield {"event": "completed", "kernel_ref": "mock_user/kernel-automl"}
            
        mock_kaggle_mgr.upload_preprocessed_splits = mock_upload
        mock_kaggle_mgr.trigger_kernel = mock_trigger
        
        # Mock synchronous polling and download methods
        mock_kaggle_mgr.get_status.return_value = "complete"
        mock_kaggle_mgr.download_outputs.return_value = None
        
        # Mock files download
        import tempfile
        from pathlib import Path
        import json
        import joblib
        
        # We need mock files downloads in downloads directory
        os.makedirs("downloads", exist_ok=True)
        best_model_path = Path("downloads/best_model.pkl")
        metrics_path = Path("downloads/metrics.json")
        
        # Save temporary files - use picklable dict instead of MagicMock
        dummy_model = {"model_name": "Random Forest Classifier", "state": "trained"}
        joblib.dump(dummy_model, best_model_path)
        with open(metrics_path, "w") as f:
            json.dump({
                "leaderboard": [{
                    "model_name": "Random Forest Classifier",
                    "fit_time": 0.05,
                    "train_metrics": {"accuracy": 1.0, "f1": 1.0},
                    "val_metrics": {"accuracy": 1.0, "f1": 1.0},
                    "cv_mean": 0.95,
                    "cv_std": 0.02,
                    "generalization_gap": 0.0,
                    "composite_score": 0.98,
                    "status_indicator": "Stable",
                    "labels": ["Stable"]
                }],
                "model_status": {"Random Forest Classifier": "completed"},
                "model_errors": {}
            }, f)
            
        # Mock Training Engine helper
        mock_engine = MagicMock()
        mock_engine.extract_feature_importance.return_value = [
            {"feature": "col_a", "importance": 0.7}
        ]
        mock_engine_cls.return_value = mock_engine
        
        # Put sample dataset in store
        _store["test_id"] = pd.DataFrame({
            "col_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        _store_filename["test_id"] = "test.csv"
        _store_ext["test_id"] = ".csv"
        
        train_payload = {
            "dataset_id": "test_id",
            "target_col": "target",
            "task_type": "binary",
            "enable_deep_optimize": False
        }
        
        res = client.post("/api/automl/train", json=train_payload)
        assert res.status_code == 200
        
        events = []
        for line in res.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                
        # Assertions to ensure event sequence and ordering
        event_types = [e.get("event") or "progress" for e in events]
        
        url_idx = -1
        upload_progress_indices = []
        kernel_progress_indices = []
        
        for idx, e in enumerate(events):
            if e.get("event") == "kaggle_url":
                url_idx = idx
            elif e.get("step") == "Kaggle Upload":
                upload_progress_indices.append(idx)
            elif e.get("step") == "Kaggle Kernel":
                kernel_progress_indices.append(idx)
                
        # Kaggle Upload progress event must be dispatched before kaggle_url event
        assert len(upload_progress_indices) > 0
        assert url_idx != -1
        assert max(upload_progress_indices) < url_idx
        
        # Clean up temporary download files
        if best_model_path.exists():
            os.remove(best_model_path)
        if metrics_path.exists():
            os.remove(metrics_path)
