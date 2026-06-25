"""
core/kaggle_client.py - Integration module for offloading AutoML workloads to Kaggle.
Handles authentication, dataset creation, kernel initialization, and progress polling.
"""

import os
import json
import time
import shutil
import tempfile
import threading
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Callable, Optional

# Setup Terminal Logger with exact formatting: timestamp, level, name, and message
logger = logging.getLogger("SOL.KaggleMLOps")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Dynamic import with fallback error message and auto-installation
KAGGLE_AVAILABLE = False
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    logger.warning("Kaggle package not found. Attempting automatic installation via pip...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        from kaggle.api.kaggle_api_extended import KaggleApi
        KAGGLE_AVAILABLE = True
        logger.info("Kaggle package successfully installed and imported.")
    except Exception as e:
        logger.error("Failed to automatically install the Kaggle package: %s", str(e))

class KaggleWorkflowManager:
    """
    Manages the lifecycle of a remote training task on Kaggle's infrastructure.
    """
    def __init__(self, username: str, api_token: str):
        if not KAGGLE_AVAILABLE:
            raise ImportError(
                "The 'kaggle' package is not installed. Please run: pip install kaggle"
            )
        self.username = username
        self.api_token = api_token
        logger.info("Initializing KaggleWorkflowManager for user: '%s'", self.username)
        self.api = self._authenticate()

    def _authenticate(self) -> "KaggleApi":
        """Authenticates with the Kaggle API using custom token."""
        logger.info("Authenticating with Kaggle API via token-based workflow...")
        
        # Clean up legacy environment variables to prevent conflicts
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Remove legacy kaggle.json to prevent authentication overrides
        legacy_json = kaggle_dir / "kaggle.json"
        if legacy_json.exists():
            try:
                legacy_json.unlink()
                logger.info("Removed legacy kaggle.json file to avoid auth conflicts.")
            except Exception as e:
                logger.warning("Could not delete legacy kaggle.json: %s", str(e))
                
        access_token_path = kaggle_dir / "access_token"
        logger.info("Writing authentication token directly to '%s'...", access_token_path)
        with open(access_token_path, "w", encoding="utf-8") as f:
            f.write(self.api_token.strip())
            
        try:
            os.chmod(access_token_path, 0o600)
            logger.info("Set 0600 secure file permissions on access_token")
        except Exception as e:
            logger.warning("Could not set strict file permissions: %s (Ignored on Windows)", str(e))
            
        # Set token environment variable for Kaggle API client
        os.environ["KAGGLE_API_TOKEN"] = self.api_token.strip()
        
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API token authentication successful. Session established.")
        return api

    def upload_preprocessed_splits(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_slug: str,
        title: str
    ) -> str:
        """
        Serializes and uploads preprocessed training/testing splits to Kaggle as a private dataset.
        
        Returns:
            str: Full dataset path (e.g. 'username/dataset-slug')
        """
        full_ref = f"{self.username}/{dataset_slug}"
        logger.info("Preparing data splits for upload to Kaggle: '%s'", full_ref)
        logger.info("Data Dimensions: Train shape = %s, Test shape = %s", X_train.shape, X_test.shape)
        
        temp_dir = Path(tempfile.mkdtemp())
        logger.info("Created temporary serialization directory: '%s'", temp_dir)
        try:
            # Write splits to temp directory
            X_train.to_csv(temp_dir / "X_train.csv", index=False)
            pd.Series(y_train).to_csv(temp_dir / "y_train.csv", index=False)
            X_test.to_csv(temp_dir / "X_test.csv", index=False)
            pd.Series(y_test).to_csv(temp_dir / "y_test.csv", index=False)
            logger.info("Successfully serialized splits to CSV.")

            # Metadata JSON
            meta = {
                "title": title,
                "id": full_ref,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(temp_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
            logger.info("Generated dataset-metadata.json descriptor.")

            # Check if dataset already exists
            logger.info("Querying existing user datasets to determine versioning path...")
            existing = [str(d) for d in self.api.dataset_list(user=self.username)]
            if any(full_ref in d for d in existing):
                logger.info("Existing dataset found. Pushing new dataset version to Kaggle...")
                self.api.dataset_create_version(
                    folder=str(temp_dir),
                    version_notes="Automatic AutoML training splits update.",
                    quiet=True
                )
                logger.info("Dataset version updated successfully.")
            else:
                logger.info("No matching dataset found. Pushing new private dataset to Kaggle...")
                self.api.dataset_create_new(
                    folder=str(temp_dir),
                    public=False,  # Force privacy
                    quiet=True
                )
                logger.info("New private dataset created successfully on Kaggle.")

            # Wait for dataset to be processed and ready on Kaggle
            logger.info("Waiting for Kaggle to process the dataset and mark it as 'ready'...")
            max_wait_seconds = 300  # 5 minutes
            poll_interval = 5
            elapsed = 0
            
            while elapsed < max_wait_seconds:
                try:
                    status_data = self.api.dataset_status(full_ref)
                    if hasattr(status_data, "status"):
                        status_val = getattr(status_data, "status")
                    else:
                        status_val = status_data
                        
                    if hasattr(status_val, "value"):
                        status = str(status_val.value).lower().strip()
                    elif hasattr(status_val, "name"):
                        status = str(status_val.name).lower().strip()
                    else:
                        status = str(status_val).lower().strip()
                        
                    logger.info("Polling dataset status for '%s': %s (elapsed: %ds)", full_ref, status.upper(), elapsed)
                    if status == "ready":
                        logger.info("Dataset '%s' is ready. Proceeding to trigger kernel.", full_ref)
                        break
                except Exception as e:
                    logger.warning("Error checking dataset status: %s (will retry)", str(e))
                
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                logger.warning("Dataset processing timed out on Kaggle (5 minutes), proceeding anyway...")

            return full_ref
        finally:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary serialization directory.")

    def trigger_kernel(
        self,
        dataset_ref: str,
        kernel_slug: str,
        task_type: str,
        models: list
    ) -> str:
        """
        Creates a training script, configures metadata, and triggers a Kaggle kernel execution.
        
        Returns:
            str: Full kernel path (e.g. 'username/kernel-slug')
        """
        kernel_ref = f"{self.username}/{kernel_slug}"
        logger.info("Preparing AutoML training script and kernel metadata for '%s'...", kernel_ref)
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Remote AutoML script runs on preprocessed data directly
            script_content = f"""
import pandas as pd
import numpy as np
import os
import joblib
import json
import glob
import time
import gc

# Standard sklearn models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from concurrent.futures import ThreadPoolExecutor

# Advanced Models (Imported with fallback in case they fail)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

print("=== REMOTE KAGGLE AUTOML PIPELINE STARTING ===")

# Mount propagation defense loop
print("Listing /kaggle/input:")
for root, dirs, files in os.walk("/kaggle/input"):
    print(f"  Root: {{root}}, Dirs: {{dirs}}, Files: {{files}}")

csv_files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
print("Found CSV files:", csv_files)

X_train_path = None
for f in csv_files:
    if os.path.basename(f) == "X_train.csv":
        X_train_path = f
        break

if not X_train_path:
    print("X_train.csv not found yet. Waiting for mount propagation...")
    for attempt in range(6):
        time.sleep(10)
        csv_files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        print(f"Attempt {{attempt+1}} - Found CSV files:", csv_files)
        for f in csv_files:
            if os.path.basename(f) == "X_train.csv":
                X_train_path = f
                break
        if X_train_path:
            break

if not X_train_path:
    raise FileNotFoundError("Could not locate X_train.csv under /kaggle/input after waiting.")

input_dir = os.path.dirname(X_train_path)
print(f"Discovered input directory: {{input_dir}}")

# Load datasets
X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).iloc[:, 0].values
X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).iloc[:, 0].values

task_type = "{task_type}"
is_cls = task_type in ["binary", "multiclass"]

# Target Encoding, Class Count Detection, and Imbalance Analysis
num_class = 2
if is_cls:
    from sklearn.preprocessing import LabelEncoder
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    if y_train_series.dtype == object or isinstance(y_train_series.iloc[0], str):
        y_train_series = y_train_series.fillna("missing").astype(str)
        y_test_series = y_test_series.fillna("missing").astype(str)
    else:
        mode_val = y_train_series.mode().iloc[0] if not y_train_series.mode().empty else 0
        y_train_series = y_train_series.fillna(mode_val).astype(int)
        y_test_series = y_test_series.fillna(mode_val).astype(int)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_series)
    y_test = le.transform(y_test_series)
    num_class = len(le.classes_)
    if num_class > 2:
        task_type = "multiclass"
    else:
        task_type = "binary"

    # --- Class Imbalance Analysis ---
    # Compute class weights using sklearn's balanced formula:
    #   w_c = n_samples / (n_classes * n_samples_in_class_c)
    # This correctly up-weights the minority class (Churn=1) relative
    # to the majority class (Non-Churn=0) for all tree-based models.
    from sklearn.utils.class_weight import compute_class_weight
    classes_array = np.unique(y_train)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_array,
        y=y_train
    )
    CLASS_WEIGHT_DICT = dict(zip(classes_array.tolist(), raw_weights.tolist()))

    # Imbalance ratio: majority_count / minority_count.
    # We only activate balancing when ratio > 1.5x to avoid penalizing
    # well-balanced datasets with unnecessary regularization.
    class_counts = np.bincount(y_train.astype(int))
    IMBALANCE_RATIO = float(class_counts.max()) / float(class_counts.min())
    USE_CLASS_WEIGHT = IMBALANCE_RATIO > 1.5

    # XGBoost binary uses scale_pos_weight = count(neg) / count(pos)
    # rather than a weight dict. Assumes class 0 = negative, class 1 = positive.
    if num_class == 2 and USE_CLASS_WEIGHT:
        SCALE_POS_WEIGHT = float(class_counts[0]) / float(class_counts[1])
    else:
        SCALE_POS_WEIGHT = 1.0

    print(
        f"Class distribution: {{dict(zip(classes_array, class_counts))}} | "
        f"Imbalance ratio: {{IMBALANCE_RATIO:.2f}}x | "
        f"Balancing active: {{USE_CLASS_WEIGHT}} | "
        f"Weights: {{{{k: round(v, 4) for k, v in CLASS_WEIGHT_DICT.items()}}}}"
    )
else:
    CLASS_WEIGHT_DICT = None
    USE_CLASS_WEIGHT = False
    IMBALANCE_RATIO = 1.0
    SCALE_POS_WEIGHT = 1.0

print(f"Train Shape: {{X_train.shape}} | Task: {{task_type}} | Classes: {{num_class}}")

# Instantiate models
# `USE_CLASS_WEIGHT`, `CLASS_WEIGHT_DICT`, and `SCALE_POS_WEIGHT` are computed
# from the training labels above and injected here to handle class imbalance.
# They are module-level globals within the remote script so both the main
# model AND each CV fold model receive identical balancing parameters.
def instantiate_model(model_name):
    # Resolve effective class weight for sklearn-compatible models.
    # Pass None (no balancing) when the dataset is balanced or for regression.
    cw = CLASS_WEIGHT_DICT if USE_CLASS_WEIGHT else None

    if model_name == "LogisticRegression":
        # LogisticRegression already uses L2 regularization, but balancing
        # still helps recall on the minority class.
        return LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight=cw
        )
    elif model_name == "LinearRegression":
        return LinearRegression()
    elif model_name == "RandomForest":
        if is_cls:
            return RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1,
                class_weight=cw
            )
        else:
            return RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
            )
    elif model_name == "ExtraTrees":
        if is_cls:
            return ExtraTreesClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1,
                class_weight=cw
            )
        else:
            return ExtraTreesRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
            )
    elif model_name == "DecisionTree":
        if is_cls:
            return DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=4, random_state=42,
                class_weight=cw
            )
        else:
            return DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=42)
    elif model_name == "GradientBoosting":
        # sklearn GradientBoosting does not support class_weight.
        # Sample weights would be needed for proper balancing, but that
        # requires changing fit() calls — deferred per A3 decision.
        if is_cls:
            return GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
    elif model_name == "KNN":
        # KNN has no class_weight parameter — inherently distance-based.
        if is_cls:
            return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        else:
            return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    elif model_name == "SVM":
        if is_cls:
            return SVC(
                probability=True, random_state=42,
                class_weight=cw
            )
        else:
            return SVR()
    elif model_name == "XGBoost" and XGB_AVAILABLE:
        if is_cls:
            if num_class > 2:
                # Multiclass: XGBoost does not support scale_pos_weight for
                # multiclass. class_weight via sample_weight is deferred (A3).
                return XGBClassifier(
                    objective="multi:softprob", num_class=num_class,
                    n_estimators=100, max_depth=5, min_child_weight=4,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0
                )
            else:
                # Binary: scale_pos_weight = count(neg) / count(pos)
                # This is XGBoost's native mechanism for imbalanced binary tasks.
                return XGBClassifier(
                    objective="binary:logistic",
                    scale_pos_weight=SCALE_POS_WEIGHT if USE_CLASS_WEIGHT else 1.0,
                    n_estimators=100, max_depth=5, min_child_weight=4,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0
                )
        else:
            return XGBRegressor(
                n_estimators=100, max_depth=5, min_child_weight=4,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0
            )
    elif model_name == "LightGBM" and LGBM_AVAILABLE:
        if is_cls:
            if num_class > 2:
                return LGBMClassifier(
                    objective="multiclass", num_class=num_class,
                    n_estimators=100, max_depth=5, num_leaves=31, min_child_samples=10,
                    subsample=0.8, colsample_bytree=0.8,
                    class_weight=cw, random_state=42, n_jobs=-1, verbose=-1
                )
            else:
                return LGBMClassifier(
                    objective="binary",
                    n_estimators=100, max_depth=5, num_leaves=31, min_child_samples=10,
                    subsample=0.8, colsample_bytree=0.8,
                    class_weight=cw, random_state=42, n_jobs=-1, verbose=-1
                )
        else:
            return LGBMRegressor(
                n_estimators=100, max_depth=5, num_leaves=31, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1
            )
    elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
        if is_cls:
            # CatBoost accepts class_weights as an ordered list indexed by
            # class label. We build it from the dict sorted by class index.
            cb_weights = (
                [CLASS_WEIGHT_DICT[c] for c in sorted(CLASS_WEIGHT_DICT.keys())]
                if USE_CLASS_WEIGHT else None
            )
            if num_class > 2:
                return CatBoostClassifier(
                    loss_function="MultiClass", iterations=100,
                    random_state=42, verbose=0, thread_count=-1,
                    class_weights=cb_weights
                )
            else:
                return CatBoostClassifier(
                    loss_function="Logloss", iterations=100,
                    random_state=42, verbose=0, thread_count=-1,
                    class_weights=cb_weights
                )
        else:
            return CatBoostRegressor(
                iterations=100, random_state=42, verbose=0, thread_count=-1
            )
    raise ValueError(f"Model {{model_name}} is not supported or not installed.")


# Evaluate helper
def evaluate_model_on_data(model, X, y):
    preds = model.predict(X)
    if is_cls:
        acc = float(accuracy_score(y, preds))
        avg_method = "binary" if num_class == 2 else "weighted"
        prec = float(precision_score(y, preds, average=avg_method, zero_division=0))
        rec = float(recall_score(y, preds, average=avg_method, zero_division=0))
        f1 = float(f1_score(y, preds, average=avg_method, zero_division=0))
        roc_auc = 0.5
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                if num_class == 2:
                    if probs.shape[1] == 2:
                        roc_auc = float(roc_auc_score(y, probs[:, 1]))
                    else:
                        roc_auc = float(roc_auc_score(y, probs))
                else:
                    roc_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="weighted"))
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                if num_class == 2:
                    roc_auc = float(roc_auc_score(y, scores))
                else:
                    roc_auc = float(roc_auc_score(y, scores, multi_class="ovr", average="weighted"))
        except Exception:
            pass
        return {{
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc
        }}
    else:
        rmse = float(root_mean_squared_error(y, preds))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))
        return {{
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }}

# Balanced composite score helper
def compute_balanced_composite(train_metrics, val_metrics, cv_scores, model_name):
    metric_key = "f1" if is_cls else "r2"
    s_train = train_metrics.get(metric_key, 0.0)
    s_val = val_metrics.get(metric_key, 0.0)
    s_train_c = max(0.0, min(1.0, s_train))
    s_val_c = max(0.0, min(1.0, s_val))
    cv_mean = float(np.mean(cv_scores)) if cv_scores else s_val_c
    cv_std = float(np.std(cv_scores)) if cv_scores else 0.0
    cv_mean_c = max(0.0, min(1.0, cv_mean))
    
    gen_gap = max(0.0, s_train_c - s_val_c)
    if gen_gap <= 0.05:
        overfit_penalty = 0.0
    elif gen_gap <= 0.15:
        overfit_penalty = 0.5 * (gen_gap - 0.05)
    else:
        overfit_penalty = 1.0 * (gen_gap - 0.05) + 0.10
        
    perfect_score_penalty = 0.0
    if s_val == 1.0 and s_train == 1.0:
        if cv_mean < 0.95:
            perfect_score_penalty = 0.20
        else:
            perfect_score_penalty = 0.05
            
    stability = max(0.0, 1.0 - 5.0 * cv_std)
    complexity_penalty = 0.0
    if model_name in ["RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]:
        complexity_penalty = 0.015
    elif model_name in ["SVM"]:
        complexity_penalty = 0.01
        
    composite_score = (
        0.4 * s_val_c + 
        0.4 * cv_mean_c + 
        0.2 * stability - 
        overfit_penalty - 
        perfect_score_penalty - 
        complexity_penalty
    )
    composite_score = float(max(0.0, min(1.0, composite_score)))
    
    labels = []
    if gen_gap > 0.15:
        labels.append("Heavy Overfitting")
        status_indicator = "Heavy Overfitting"
    elif gen_gap > 0.05:
        labels.append("Slight Overfitting")
        status_indicator = "Slight Overfitting"
    elif s_train_c < 0.55:
        labels.append("Underfitting")
        status_indicator = "Underfitting"
    else:
        labels.append("Stable")
        status_indicator = "Stable"
        
    if cv_std > 0.08:
        labels.append("High Variance")
    if "Stable" in labels and cv_std < 0.04 and composite_score >= 0.70:
        labels.append("Recommended")
    else:
        labels.append("Experimental")
        
    return composite_score, gen_gap, cv_std, status_indicator, labels

# List of models to evaluate
models_to_train = {models}
print("Models scheduled for training:", models_to_train)

leaderboard = []
trained_instances = {{}}
model_status = {{}}
model_errors = {{}}

# K-Fold CV splits
cv_folds = 5
total_cells = X_train.shape[0] * X_train.shape[1]
if total_cells > 200000:
    cv_folds = 3

for name in models_to_train:
    print(f"Training {{name}}...")
    t0 = time.time()
    try:
        model = instantiate_model(name)
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        train_metrics = evaluate_model_on_data(model, X_train, y_train)
        val_metrics = evaluate_model_on_data(model, X_test, y_test)
        
        cv_splitter = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            if is_cls else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        )
        metric_key = "f1" if is_cls else "r2"
        
        def train_cv_fold_baseline(train_idx, val_idx):
            X_tr_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
            
            cv_model = instantiate_model(name)
            try:
                if hasattr(cv_model, 'n_jobs'):
                    cv_model.set_params(n_jobs=1)
            except Exception:
                pass
            try:
                if hasattr(cv_model, 'thread_count'):
                    cv_model.set_params(thread_count=1)
            except Exception:
                pass
            try:
                if hasattr(cv_model, 'nthread'):
                    cv_model.set_params(nthread=1)
            except Exception:
                pass
            cv_model.fit(X_tr_cv, y_tr_cv)
            cv_evals = evaluate_model_on_data(cv_model, X_val_cv, y_val_cv)
            return cv_evals.get(metric_key, 0.0)

        with ThreadPoolExecutor(max_workers=max(8, cv_folds)) as executor:
            cv_scores = list(executor.map(
                lambda split: train_cv_fold_baseline(split[0], split[1]),
                cv_splitter.split(X_train, y_train)
            ))
            
        composite_score, gen_gap, cv_std, status_indicator, labels = compute_balanced_composite(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            cv_scores=cv_scores,
            model_name=name
        )
        
        res = {{
            "model_name": name,
            "fit_time": fit_time,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
            "cv_std": cv_std,
            "generalization_gap": gen_gap,
            "composite_score": composite_score,
            "status_indicator": status_indicator,
            "labels": labels
        }}
        
        leaderboard.append(res)
        trained_instances[name] = model
        model_status[name] = "completed"
        print(f"  --> {{name}} Completed | Composite Score: {{composite_score:.4f}}")
    except Exception as e:
        import traceback
        err_msg = str(e) + "\\n" + traceback.format_exc()
        print(f"  --> {{name}} Failed: {{err_msg}}")
        model_status[name] = "failed"
        model_errors[name] = err_msg
    finally:
        gc.collect()

# Sort leaderboard descending by composite score
leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)

out_dir = "/kaggle/working"
os.makedirs(out_dir, exist_ok=True)

if leaderboard:
    # Slice the top 3 baseline models
    top_3_baselines = leaderboard[:3]
    print(f"Top 3 baseline models selected for deep tuning: {{[m['model_name'] for m in top_3_baselines]}}")
    
    tuned_models_res = []
    
    for baseline_res in top_3_baselines:
        model_name = baseline_res["model_name"]
        print(f"Executing deep tuning sweep for: {{model_name}}...")
        
        # Remote Hyperparameter Tuning (Path B HPO)
        param_dists = {{}}
        if model_name == "RandomForest":
            param_dists = {{
                "n_estimators": [50, 100, 150],
                "max_depth": [6, 10, 15],
                "min_samples_leaf": [2, 4, 8],
                "min_samples_split": [5, 10]
            }}
        elif model_name == "ExtraTrees":
            param_dists = {{
                "n_estimators": [50, 100, 150],
                "max_depth": [6, 10, 15],
                "min_samples_leaf": [2, 4, 8],
                "min_samples_split": [5, 10]
            }}
        elif model_name == "DecisionTree":
            param_dists = {{
                "max_depth": [4, 6, 8, 10],
                "min_samples_leaf": [2, 4, 8],
                "min_samples_split": [5, 10]
            }}
        elif model_name == "LogisticRegression":
            param_dists = {{
                "C": [0.01, 0.1, 1.0, 10.0]
            }}
        elif model_name == "SVM":
            param_dists = {{
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"]
            }}
        elif model_name == "GradientBoosting":
            param_dists = {{
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5]
            }}
        elif model_name == "XGBoost" and XGB_AVAILABLE:
            param_dists = {{
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6]
            }}
        elif model_name == "LightGBM" and LGBM_AVAILABLE:
            param_dists = {{
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [15, 31]
            }}
        elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
            param_dists = {{
                "iterations": [50, 100],
                "learning_rate": [0.01, 0.1]
            }}

        if param_dists:
            from sklearn.model_selection import RandomizedSearchCV
            scoring = "f1_weighted" if is_cls else "neg_root_mean_squared_error"
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_cls else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            base_search_estimator = instantiate_model(model_name)
            search = RandomizedSearchCV(
                estimator=base_search_estimator,
                param_distributions=param_dists,
                n_iter=5,
                cv=cv,
                scoring=scoring,
                random_state=42,
                n_jobs=-1
            )
            try:
                t_tune_0 = time.time()
                search.fit(X_train, y_train)
                tuned_model = search.best_estimator_
                tune_time = time.time() - t_tune_0
                print(f"  Tuning complete for {{model_name}}. Best params: {{search.best_params_}}")
                
                # Evaluate tuned model
                train_metrics_opt = evaluate_model_on_data(tuned_model, X_train, y_train)
                val_metrics_opt = evaluate_model_on_data(tuned_model, X_test, y_test)
                
                # CV evaluations parallelized
                def train_cv_fold_opt_fn(train_idx_fold, val_idx_fold):
                    X_tr_cv, X_val_cv = X_train.iloc[train_idx_fold], X_train.iloc[val_idx_fold]
                    y_tr_cv, y_val_cv = y_train[train_idx_fold], y_train[val_idx_fold]
                    
                    cv_model = instantiate_model(model_name)
                    cv_model.set_params(**search.best_params_)
                    
                    # Control inner parallelism of base estimators to prevent CPU thrashing/oversubscription
                    try:
                        if hasattr(cv_model, 'n_jobs'):
                            cv_model.set_params(n_jobs=1)
                    except Exception:
                        pass
                    try:
                        if hasattr(cv_model, 'thread_count'):
                            cv_model.set_params(thread_count=1)
                    except Exception:
                        pass
                    try:
                        if hasattr(cv_model, 'nthread'):
                            cv_model.set_params(nthread=1)
                    except Exception:
                        pass
                        
                    cv_model.fit(X_tr_cv, y_tr_cv)
                    cv_evals = evaluate_model_on_data(cv_model, X_val_cv, y_val_cv)
                    return cv_evals.get(metric_key, 0.0)

                with ThreadPoolExecutor(max_workers=max(8, cv_folds)) as executor:
                    cv_scores_opt = list(executor.map(
                        lambda split: train_cv_fold_opt_fn(split[0], split[1]),
                        cv.split(X_train, y_train)
                    ))
                    
                composite_score_opt, gen_gap_opt, cv_std_opt, status_indicator_opt, labels_opt = compute_balanced_composite(
                    train_metrics=train_metrics_opt,
                    val_metrics=val_metrics_opt,
                    cv_scores=cv_scores_opt,
                    model_name=model_name
                )
                
                tuned_res = {{
                    "model_name": f"Tuned_{{model_name}}",
                    "fit_time": baseline_res["fit_time"] + tune_time,
                    "train_metrics": train_metrics_opt,
                    "val_metrics": val_metrics_opt,
                    "cv_mean": float(np.mean(cv_scores_opt)) if cv_scores_opt else 0.0,
                    "cv_std": cv_std_opt,
                    "generalization_gap": gen_gap_opt,
                    "composite_score": composite_score_opt,
                    "status_indicator": status_indicator_opt,
                    "labels": labels_opt
                }}
                
                tuned_name = f"Tuned_{{model_name}}"
                trained_instances[tuned_name] = tuned_model
                model_status[tuned_name] = "completed"
                tuned_models_res.append(tuned_res)
                print(f"  --> {{tuned_name}} Completed | Composite Score: {{composite_score_opt:.4f}}")
            except Exception as e_opt:
                print(f"  --> Tuning failed for {{model_name}}: {{str(e_opt)}}")
                
    # Merge tuned results and sort
    leaderboard.extend(tuned_models_res)
    leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Save the ultimate champion model
    best_model_name = leaderboard[0]["model_name"]
    best_model = trained_instances[best_model_name]
    best_model_path = os.path.join(out_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Ultimate Champion model '{{best_model_name}}' saved successfully to {{best_model_path}}.")
    
    # Save detailed metrics of all models
    metrics_data = {{
        "leaderboard": leaderboard,
        "model_status": model_status,
        "model_errors": model_errors
    }}
    
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Detailed metrics saved successfully to {{metrics_path}}.")
else:
    print("Error: No models were successfully trained.")
"""
            script_path = temp_dir / "remote_training.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            logger.info("Wrote training python code file remote_training.py")

            # Metadata settings
            metadata = {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": "remote_training.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset_ref],
                "kernel_sources": [],
                "competition_sources": []
            }
            with open(temp_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Generated kernel-metadata.json container configurations.")

            logger.info("Pushing kernel code and starting execution on Kaggle Sandbox...")
            self.api.kernels_push(str(temp_dir))
            logger.info("Kernel successfully pushed. Remote run initiated.")
            return kernel_ref
        finally:
            shutil.rmtree(temp_dir)

    def get_status(self, kernel_ref: str) -> str:
        """Returns current kernel state: 'running', 'complete', or 'error'."""
        try:
            status_data = self.api.kernels_status(kernel_ref)
            if isinstance(status_data, dict):
                status = status_data.get("status", "error")
            else:
                status = getattr(status_data, "status", "error")
                
            # Resolve status to a lowercase string or numeric representation safely
            if hasattr(status, "value"):
                status_val = status.value
            elif hasattr(status, "name"):
                status_val = status.name
            else:
                status_val = status
                
            # Map status code (can be string, enum, or numeric like 0, 1, 3, 4)
            # 0: queued, 1: running, 3: complete, 4: error
            status_str = str(status_val).lower().strip()
            
            if status_str == "complete" or status_str in ["2", "3"] or "complete" in status_str:
                mapped_status = "complete"
            elif status_str in ("4", "error", "failed"):
                mapped_status = "error"
            elif status_str in ("1", "running"):
                mapped_status = "running"
            elif status_str in ("0", "queued"):
                mapped_status = "queued"
            else:
                mapped_status = status_str
                
            logger.info("Remote kernel status check for '%s': %s (raw: %s)", kernel_ref, mapped_status.upper(), status_val)
            return mapped_status
        except Exception as e:
            logger.error("Error querying remote status: %s", str(e))
            return "error"

    def download_outputs(self, kernel_ref: str, dest_dir: Path):
        """Downloads all working directory output files from the completed kernel."""
        logger.info("Downloading training output artifacts from completed Kaggle kernel '%s'...", kernel_ref)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old files to prevent loading stale models on failure or silent skip
        for filename in ["best_model.pkl", "metrics.json"]:
            filepath = dest_dir / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                    logger.info("Cleaned stale file: %s", filepath)
                except Exception as e:
                    logger.warning("Could not delete stale file %s: %s", filepath, str(e))
                    
        self.api.kernels_output(kernel_ref, path=str(dest_dir), force=True)
        logger.info("Outputs successfully downloaded and stored in: '%s'", dest_dir)
