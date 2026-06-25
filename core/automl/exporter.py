import os
import json
import joblib
import zipfile
import io
import time
import pandas as pd
from typing import Dict, List, Any, Optional

class AutoMLArtifactExporter:
    """
    Handles premium packaging and nested serialization of SOL AutoML artifacts.
    """
    
    @staticmethod
    def serialize_to_zip(
        best_model: Any,
        preprocessor: Any,
        target_encoder: Optional[Any],
        metrics: Dict[str, Any],
        col_types: Dict[str, List[str]],
        task_type: str,
        target_col: str,
        dataset_shape: tuple,
        best_model_name: str,
        feature_importance: List[Dict[str, Any]],
        original_df: pd.DataFrame,
        pdf_report_bytes: bytes,
        visualizations_dict: Dict[str, bytes],  # key -> png bytes
        failed_models_log: Dict[str, str],      # model -> traceback
        training_logs: str,
        timings_dict: Dict[str, float]
    ) -> io.BytesIO:
        # Coerce inputs safely to string
        task_type = str(task_type) if task_type else "classification"
        target_col = str(target_col) if target_col else "target"
        best_model_name = str(best_model_name) if best_model_name else "model"

        # Coerce feature_importance if passed as a dictionary
        if isinstance(feature_importance, dict):
            feature_importance = [
                {"feature": f, "importance": float(i)}
                for f, i in feature_importance.items()
            ]
        elif not feature_importance:
            feature_importance = []

        zip_buffer = io.BytesIO()
        
        # Build dataset base name
        dataset_name = "dataset"
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Prefix for all internal files to create a clean root folder inside zip
            prefix = f"SOL AutoML - {dataset_name}/"
            
            # 1. Best Model
            model_bytes = io.BytesIO()
            joblib.dump(best_model, model_bytes)
            model_bytes.seek(0)
            zip_file.writestr(f"{prefix}trained_model.pkl", model_bytes.getvalue())
            
            # 2. Preprocessing Pipeline Bundle
            num_features = [c for c in col_types["numerical"] if c != target_col]
            cat_features = [c for c in col_types["categorical"] if c != target_col]
            
            prep_bundle = {
                "preprocessor": preprocessor,
                "target_encoder": target_encoder,
                "numerical_cols": num_features,
                "categorical_cols": cat_features
            }
            prep_bytes = io.BytesIO()
            joblib.dump(prep_bundle, prep_bytes)
            prep_bytes.seek(0)
            zip_file.writestr(f"{prefix}preprocessing_pipeline.pkl", prep_bytes.getvalue())
            
            # 3. Metrics JSON
            metrics_str = json.dumps(metrics, indent=4)
            zip_file.writestr(f"{prefix}metrics.json", metrics_str)
            
            # 4. Feature Columns JSON
            feat_cols = {
                "numerical": col_types.get("numerical", []),
                "categorical": col_types.get("categorical", []),
                "datetime": col_types.get("datetime", []),
                "target": target_col
            }
            feat_cols_str = json.dumps(feat_cols, indent=4)
            zip_file.writestr(f"{prefix}feature_columns.json", feat_cols_str)
            
            # 5. Training Summary JSON
            summary = {
                "system_name": "SOL AutoML System",
                "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_shape": {
                    "rows": dataset_shape[0],
                    "columns": dataset_shape[1]
                },
                "ml_task": task_type,
                "best_model": best_model_name,
                "target_column": target_col,
                "evaluated_metrics": metrics,
                "feature_importance_top_10": feature_importance[:10]
            }
            summary_str = json.dumps(summary, indent=4)
            zip_file.writestr(f"{prefix}training_summary.json", summary_str)
            
            # 6. PDF Executive Report
            zip_file.writestr(f"{prefix}report.pdf", pdf_report_bytes)
            
            # 7. Test Sample CSV (5 random rows with target column removed)
            try:
                sample_df = original_df.sample(n=min(5, len(original_df)), random_state=42)
                # Drop target column if exists
                if target_col in sample_df.columns:
                    sample_df = sample_df.drop(columns=[target_col])
                csv_buffer = io.StringIO()
                sample_df.to_csv(csv_buffer, index=False)
                zip_file.writestr(f"{prefix}test_sample.csv", csv_buffer.getvalue())
            except Exception as csv_err:
                print(f"Failed to generate test sample csv: {csv_err}")
                # Fallback to simple structure
                zip_file.writestr(f"{prefix}test_sample.csv", "No samples generated due to data-slicing constraint.")
                
            # 8. Visualization Subdirectory (Singular: "visualization" as requested)
            for chart_name, chart_bytes in visualizations_dict.items():
                if chart_bytes:
                    zip_file.writestr(f"{prefix}visualization/{chart_name}.png", chart_bytes)
                    
            # 9. Logs Subdirectory
            zip_file.writestr(f"{prefix}logs/training_logs.txt", training_logs)
            
            failed_str = json.dumps(failed_models_log, indent=4)
            zip_file.writestr(f"{prefix}logs/failed_models.json", failed_str)
            
            timings_str = json.dumps(timings_dict, indent=4)
            zip_file.writestr(f"{prefix}logs/timings.json", timings_str)
            
            # 10. Standalone Deploymentpredict.py
            deploy_script = AutoMLArtifactExporter.generate_deployment_script(target_col)
            zip_file.writestr(f"{prefix}predict.py", deploy_script)
            
        zip_buffer.seek(0)
        return zip_buffer
        
    @staticmethod
    def generate_deployment_script(target_col: str) -> str:
        """
        Generates a premium deployment script showing how to load the preprocessing 
        pipeline and trained estimator for instant runtime predictions.
        """
        return f"""# -*- coding: utf-8 -*-
\"\"\"
SOL AutoML - Standalone Model Deployment Script
Usage:
    import pandas as pd
    from predict import predict
    
    predictions = predict("test_sample.csv")
    print(predictions)
\"\"\"
import joblib
import pandas as pd
import numpy as np

def load_pipeline():
    model = joblib.load("trained_model.pkl")
    prep_bundle = joblib.load("preprocessing_pipeline.pkl")
    
    preprocessor = prep_bundle["preprocessor"]
    target_encoder = prep_bundle["target_encoder"]
    numerical_cols = prep_bundle["numerical_cols"]
    categorical_cols = prep_bundle["categorical_cols"]
    
    return model, preprocessor, target_encoder, numerical_cols, categorical_cols

def predict(csv_path: str):
    # 1. Load artifacts
    model, preprocessor, target_encoder, num_cols, cat_cols = load_pipeline()
    
    # 2. Read inputs
    df = pd.read_csv(csv_path)
    
    # 3. Align features
    features = num_cols + cat_cols
    X = df[features]
    
    # 4. Transform features
    X_processed = preprocessor.transform(X)
    
    # Format to DataFrame to preserve feature alignments
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
            cleaned_names = [f.split("__")[-1] for f in feature_names]
            if isinstance(X_processed, pd.DataFrame):
                X_processed.columns = cleaned_names
            else:
                X_processed = pd.DataFrame(X_processed, columns=cleaned_names)
        elif isinstance(X_processed, pd.DataFrame):
            pass
        else:
            X_processed = pd.DataFrame(X_processed)
    except Exception:
        if not isinstance(X_processed, pd.DataFrame):
            X_processed = pd.DataFrame(X_processed)
            
    # Align strictly to model's fitted features
    expected_features = None
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "feature_name"):
        try:
            expected_features = list(model.feature_name())
        except Exception:
            pass
    elif hasattr(model, "get_booster"):
        try:
            expected_features = model.get_booster().feature_names
        except Exception:
            pass
            
    if expected_features:
        # Check duplicate features
        dup_cols = X_processed.columns[X_processed.columns.duplicated()].tolist()
        if dup_cols:
            raise ValueError(f"Feature columns contain duplicates: {{dup_cols}}")
            
        # Check missing features
        missing_cols = [c for c in expected_features if c not in X_processed.columns]
        if missing_cols:
            raise ValueError(f"Input is missing expected columns: {{missing_cols}}")
            
        X_processed = X_processed[expected_features]
        
    # 5. Predict
    predictions = model.predict(X_processed)
    
    # 6. Inverse transform target if classification
    if target_encoder is not None:
        return target_encoder.inverse_transform(predictions)
        
    return predictions

if __name__ == "__main__":
    print("SOL AutoML Standalone predict.py script loaded.")
    print("Call predict('test_sample.csv') to execute predictions.")
"""
