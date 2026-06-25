# -*- coding: utf-8 -*-
"""
SOL AutoML - Standalone Model Deployment Script
Usage:
    import pandas as pd
    from predict import predict
    
    predictions = predict("test_sample.csv")
    print(predictions)
"""
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
            raise ValueError(f"Feature columns contain duplicates: {dup_cols}")
            
        # Check missing features
        missing_cols = [c for c in expected_features if c not in X_processed.columns]
        if missing_cols:
            raise ValueError(f"Input is missing expected columns: {missing_cols}")
            
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
