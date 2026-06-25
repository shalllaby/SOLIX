import io
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from backend.main import app
from backend.tools.automl.router import _SESSIONS

client = TestClient(app)

class DummyModel:
    def predict(self, X):
        return np.array([42.0])

def test_predict_endpoint_robustness():
    # Setup a dummy session in _SESSIONS
    session_id = "test-predict-session"
    
    # Create a simple ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["col_a", "col_b"])
        ]
    )
    # Fit on dummy data
    X_dummy = pd.DataFrame({"col_a": [1.0, 2.0, 3.0], "col_b": [4.0, 5.0, 6.0]})
    preprocessor.fit(X_dummy)
    
    # Create a dummy trained model
    best_model = DummyModel()
    
    # Serialize best_model and preprocessor to bytes using joblib
    import joblib
    model_bytes = io.BytesIO()
    joblib.dump(best_model, model_bytes)
    model_bytes.seek(0)
    
    prep_bytes = io.BytesIO()
    joblib.dump(preprocessor, prep_bytes)
    prep_bytes.seek(0)
    
    _SESSIONS[session_id] = {
        "best_model_bytes": model_bytes.getvalue(),
        "preprocessor_bytes": prep_bytes.getvalue(),
        "task_type": "regression",
        "best_model_name": "DummyModel"
    }
    
    # Now call predict with empty inputs or missing features
    # (should successfully fill and predict instead of raising TypeError)
    payload = {
        "session_id": session_id,
        "input_data": json.dumps({})  # Empty JSON inputs
    }
    
    res = client.post("/api/automl/predict", data=payload)
    assert res.status_code == 200
    assert res.json()["prediction"] == "42.0"
