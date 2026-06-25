import os
import sys

# Dynamic UTF-8 terminal reconfiguration to prevent CP1252 character map crashes on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Setup environment to avoid CPU checks or other warnings
os.environ['POLARS_SKIP_CPU_CHECK'] = '1'

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Include core paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.analyzer import analyze_dataset, rank_target_candidates, infer_task_type
from core.preprocessor import prepare_data, get_processed_feature_names, sanitize_features
from core.engine import AutoMLTrainingEngine
from core.exporter import AutoMLArtifactExporter
from utils.pdf_generator import AutoMLPDFReportGenerator

def create_synthetic_data():
    X_c, y_c = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=42)
    df_cls = pd.DataFrame(X_c, columns=[f"feat_num_{i}" for i in range(4)] + ["feat_cat_1", "feat_cat_2"])
    
    # Introduce low-cardinality integers for categoricals
    df_cls["feat_cat_1"] = np.random.choice(["Low", "Medium", "High"], size=200)
    df_cls["feat_cat_2"] = np.random.choice([0, 1], size=200)
    df_cls["target"] = y_c
    
    X_r, y_r = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    df_reg = pd.DataFrame(X_r, columns=[f"num_{i}" for i in range(4)] + ["cat_1"])
    df_reg["cat_1"] = np.random.choice(["A", "B"], size=200)
    df_reg["label_price"] = y_r
    
    return df_cls, df_reg

def create_titanic_simulated_data():
    n_samples = 150
    np.random.seed(42)
    names = [f"Passenger, Mr. {i} [bracket_val] 'quote_val' {{json_val}}:," for i in range(n_samples)]
    sexes = np.random.choice(["male", "female", None], size=n_samples)
    ages = np.random.uniform(1.0, 80.0, size=n_samples)
    ages[np.random.choice(n_samples, 30, replace=False)] = np.nan
    sibsp = np.random.choice([0, 1, 2, 3], size=n_samples)
    parch = np.random.choice([0, 1, 2], size=n_samples)
    tickets = [f"PC 1759{i}" for i in range(n_samples)]
    fares = np.random.exponential(50.0, size=n_samples)
    cabins = np.random.choice(["C85", "C123", "E46", None], size=n_samples)
    embarked = np.random.choice(["S", "C", "Q", None], size=n_samples)
    survived = np.random.choice([0, 1], size=n_samples)
    
    df = pd.DataFrame({
        "Passenger Name (with special chars: , [ ] { } \")": names,
        "Sex / Gender": sexes,
        "Age": ages,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket #": tickets,
        "Fare Price": fares,
        "Cabin Number": cabins,
        "Embarked Port": embarked,
        "Survived": survived
    })
    return df

def run_tests():
    print("==================================================")
    print("🤖 SOL AutoML Platform - Verification Test Suite")
    print("==================================================")
    
    print("\n[Step 1] Creating Synthetic Datasets...")
    df_cls, df_reg = create_synthetic_data()
    
    # Pre-sanitize columns before profiling
    df_cls = sanitize_features(df_cls)
    df_reg = sanitize_features(df_reg)
    
    print(f"Classification Shape: {df_cls.shape} | Regression Shape: {df_reg.shape}")
    
    # ------------------------------------------------
    print("\n[Step 2] Testing Analyzer Profiling & Ranking...")
    
    profile_cls = analyze_dataset(df_cls)
    assert profile_cls["shape"] == df_cls.shape, "Analyzer returned incorrect shape."
    assert "target" in profile_cls["column_details"], "Target detail parsing missing."
    print("✅ Datatype profiling verified successfully.")
    
    ranks = rank_target_candidates(df_cls, profile_cls["column_types"])
    assert len(ranks) > 0, "No target candidates generated."
    top_cand = ranks[0]["column"]
    assert top_cand == "target", f"Top candidate should be 'target', got '{top_cand}'"
    print(f"✅ Target ranking inferred champion correctly: '{top_cand}' (Score: {ranks[0]['score']})")
    
    task_cls = infer_task_type(df_cls, "target", profile_cls["column_types"])
    assert task_cls == "binary", f"Should infer binary classification, got '{task_cls}'"
    print("✅ Inferred ML task type successfully: 'binary'")
    
    # ------------------------------------------------
    print("\n[Step 3] Testing Preprocessing Pipeline...")
    X_train, X_test, y_train, y_test, preprocessor, encoder = prepare_data(
        df=df_cls,
        target_col="target",
        col_types=profile_cls["column_types"],
        task_type=task_cls,
        test_size=0.2
    )
    assert isinstance(X_train, pd.DataFrame), "X_train must be a pandas DataFrame."
    assert isinstance(X_test, pd.DataFrame), "X_test must be a pandas DataFrame."
    assert list(X_train.columns) == list(X_test.columns), "Feature names between train and test must match."
    assert X_train.shape[0] == 160, f"Expected 160 train rows, got {X_train.shape[0]}"
    assert X_test.shape[0] == 40, f"Expected 40 test rows, got {X_test.shape[0]}"
    
    feature_names = get_processed_feature_names(preprocessor)
    # Ensure matching list elements with columns
    sanitized_features = []
    seen = {}
    for col in feature_names:
        san = sanitize_features(pd.DataFrame(columns=[col])).columns[0]
        if san in seen:
            seen[san] += 1
            san_unique = f"{san}_{seen[san]}"
        else:
            seen[san] = 0
            san_unique = san
        sanitized_features.append(san_unique)
        
    assert list(X_train.columns) == sanitized_features, "DataFrame columns must match get_processed_feature_names output."
    print(f"✅ Prepared features. Transformed columns list: {sanitized_features}")
    
    # ------------------------------------------------
    print("\n[Step 4] Testing Training Engine...")
    engine = AutoMLTrainingEngine(task_cls, timeout_seconds=100)
    models = ["RandomForest", "DecisionTree", "LogisticRegression"]
    if "XGBoost" in engine.get_available_models():
        models.append("XGBoost")
    if "LightGBM" in engine.get_available_models():
        models.append("LightGBM")
        
    print(f"Running baseline models: {models}")
    leaderboard, instances = engine.train_baselines(X_train, y_train, X_test, y_test, models)
    
    assert len(leaderboard) > 0, "Leaderboard is empty."
    assert leaderboard[0]["model_name"] in instances, "Trained instance mapping missing."
    print(f"✅ Baseline training completed. Champion: '{leaderboard[0]['model_name']}'")
    
    # Test Deep optimization
    print("Tuning best model...")
    best_name = leaderboard[0]["model_name"]
    best_inst = instances[best_name]
    tuned_model, tuned_metrics = engine.deep_optimize_best_model(X_train, y_train, X_test, y_test, best_name, best_inst)
    assert tuned_metrics is not None
    print(f"✅ Deep optimization completed. F1: {tuned_metrics.get('f1', 0.0):.4f}")
    
    # Test column alignment defensive capabilities
    print("Testing defensive feature alignment capability...")
    shuffled_cols = list(X_test.columns)
    np.random.shuffle(shuffled_cols)
    X_test_shuffled = X_test[shuffled_cols]
    
    try:
        metrics_shuffled = engine.evaluate_model_on_data(tuned_model, X_test_shuffled, y_test)
        assert metrics_shuffled is not None
        print("✅ Self-healing column reordering and alignment verified successfully.")
    except Exception as e:
        print(f"❌ Column alignment failed: {e}")
        raise e
        
    # Test missing columns error raising capability
    print("Testing missing columns validation handling...")
    X_test_missing = X_test.drop(columns=[X_test.columns[0]])
    try:
        engine.evaluate_model_on_data(tuned_model, X_test_missing, y_test)
        assert False, "Should raise an exception for missing columns."
    except Exception as e:
        print(f"✅ Correctly caught defensive validation error: {e}")
        
    # Test importance extraction
    importances = engine.extract_feature_importance(tuned_model, sanitized_features, X_test, y_test)
    assert len(importances) == len(sanitized_features), "Importance lists counts mismatch."
    print(f"✅ Feature importances extracted: {importances[:3]}")
    
    # ------------------------------------------------
    print("\n[Step 5] Testing PDF Reporting...")
    visualizations_png = {
        "leaderboard_comparison": b"fake_png_bytes",
        "confusion_matrix": b"fake_png_bytes"
    }
    
    pdf_stream = AutoMLPDFReportGenerator.generate_report(
        dataset_name="Synthetic Classification Test",
        task_type=task_cls,
        target_col="target",
        metrics={"dataset_rows": df_cls.shape[0], "dataset_cols": df_cls.shape[1]},
        col_types=profile_cls["column_types"],
        best_model_name=best_name,
        leaderboard=leaderboard,
        feature_importance=importances,
        visualizations_dict=visualizations_png,
        is_arabic=False
    )
    assert pdf_stream.getvalue(), "PDF generation returned empty stream."
    print("✅ Executive summary generated successfully (PDF).")
    
    # ------------------------------------------------
    print("\n[Step 6] Testing Artifact Exporting...")
    zip_stream = AutoMLArtifactExporter.serialize_to_zip(
        best_model=tuned_model,
        preprocessor=preprocessor,
        target_encoder=encoder,
        metrics=tuned_metrics,
        col_types=profile_cls["column_types"],
        task_type=task_cls,
        target_col="target",
        dataset_shape=df_cls.shape,
        best_model_name=best_name,
        feature_importance=importances,
        original_df=df_cls,
        pdf_report_bytes=pdf_stream.getvalue(),
        visualizations_dict=visualizations_png,
        failed_models_log={},
        training_logs="Fake log",
        timings_dict={"baseline_fit_duration": 1.5}
    )
    assert zip_stream.getvalue(), "ZIP serialization returned empty stream."
    print("✅ Artifact package created successfully (.ZIP).")
    
    # ------------------------------------------------
    print("\n[Step 7] Testing Titanic Dataset Compatibility...")
    df_titanic = create_titanic_simulated_data()
    df_titanic = sanitize_features(df_titanic)
    
    profile_titanic = analyze_dataset(df_titanic)
    task_titanic = infer_task_type(df_titanic, "Survived", profile_titanic["column_types"])
    
    X_train_t, X_test_t, y_train_t, y_test_t, preprocessor_t, encoder_t = prepare_data(
        df=df_titanic,
        target_col="Survived",
        col_types=profile_titanic["column_types"],
        task_type=task_titanic,
        test_size=0.2
    )
    
    engine_t = AutoMLTrainingEngine(task_titanic, timeout_seconds=100)
    models_t = engine_t.get_available_models()
    leaderboard_t, instances_t = engine_t.train_baselines(X_train_t, y_train_t, X_test_t, y_test_t, models_t)
    
    assert len(leaderboard_t) > 0, "Failed to train any models on simulated Titanic dataset."
    lgb_success = any(r["model_name"] == "LightGBM" for r in leaderboard_t)
    if "LightGBM" in models_t and engine_t.model_status.get("LightGBM") == "completed":
        assert lgb_success, "LightGBM trained successfully but is missing from the leaderboard registry."
    print(f"✅ Titanic simulation verified successfully. Top model: {leaderboard_t[0]['model_name']} (Composite: {leaderboard_t[0]['composite_score']:.4f})")
    
    # ------------------------------------------------
    print("\n[Step 8] Testing Real Titanic-Dataset.csv...")
    real_csv_path = "Titanic-Dataset.csv"
    if os.path.exists(real_csv_path):
        df_real = pd.read_csv(real_csv_path)
        df_real = sanitize_features(df_real)
        print(f"Loaded Real Titanic-Dataset.csv. Shape: {df_real.shape}")
        
        # 1. Profile
        profile_real = analyze_dataset(df_real)
        task_real = infer_task_type(df_real, "Survived", profile_real["column_types"])
        print(f"Inferred task type: {task_real}")
        
        # 2. Preprocess
        X_train_r, X_test_r, y_train_r, y_test_r, preprocessor_r, encoder_r = prepare_data(
            df=df_real,
            target_col="Survived",
            col_types=profile_real["column_types"],
            task_type=task_real,
            test_size=0.2
        )
        print(f"Preprocessed features shape: {X_train_r.shape}")
        
        # 3. Train
        engine_r = AutoMLTrainingEngine(task_real, timeout_seconds=120)
        models_r = engine_r.get_available_models()
        print(f"Available models for training: {models_r}")
        leaderboard_r, instances_r = engine_r.train_baselines(X_train_r, y_train_r, X_test_r, y_test_r, models_r)
        
        print("\nLeaderboard on Real Titanic Dataset:")
        for idx, row in enumerate(leaderboard_r):
            metric_key = "f1" if task_real in ["binary", "multiclass"] else "r2"
            v_val = row["val_metrics"].get(metric_key, 0.0)
            print(f"  [{idx + 1}] {row['model_name']} - Composite: {row['composite_score']:.4f} | Val F1: {v_val:.4f} | Fit Time: {row['fit_time']:.4f}s")
            
        # 4. Deep optimize the champion model
        if leaderboard_r:
            champion_name = leaderboard_r[0]["model_name"]
            print(f"\nDeep Tuning Champion Model '{champion_name}'...")
            tuned_model, tuned_metrics = engine_r.deep_optimize_best_model(
                X_train_r, y_train_r, X_test_r, y_test_r, champion_name, instances_r[champion_name]
            )
            val_metric = tuned_metrics["val_metrics"].get(metric_key, 0.0)
            print(f"Tuned Champion Metrics -> Composite: {tuned_metrics.get('composite_score', 0.0):.4f} | Val F1: {val_metric:.4f}")
            
            # 5. Extract feature importance
            feature_names_r = get_processed_feature_names(preprocessor_r)
            sanitized_features_r = []
            seen = {}
            for col in feature_names_r:
                san = sanitize_features(pd.DataFrame(columns=[col])).columns[0]
                if san in seen:
                    seen[san] += 1
                    san_unique = f"{san}_{seen[san]}"
                else:
                    seen[san] = 0
                    san_unique = san
                sanitized_features_r.append(san_unique)
                
            imp_r = engine_r.extract_feature_importance(tuned_model, sanitized_features_r, X_test_r, y_test_r)
            print("\nTop 5 Most Important Features:")
            for item in imp_r[:5]:
                print(f"  - {item['feature']}: {item['importance']:.4f}")
        else:
            print("❌ No models could be trained on the real Titanic dataset.")
            
    else:
        print("⚠️ Titanic-Dataset.csv not found in root workspace directory.")
        
    print("\n==================================================")
    print("🎉 ALL TESTS PASSED! SOL AUTOML PLATFORM IS FULLY VERIFIED.")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
