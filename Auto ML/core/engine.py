import time
import numpy as np
import pandas as pd
import threading
import gc
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

class AutoMLValidationError(Exception):
    """Custom exception class for defensive AutoML validation checks."""
    pass

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
from sklearn.inspection import permutation_importance

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

# CatBoost subprocess detection to bypass AVX crashes on restricted hardware
CATBOOST_AVAILABLE = False
try:
    import subprocess
    import sys
    res = subprocess.run(
        [sys.executable, "-c", "import catboost"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if res.returncode == 0:
        from catboost import CatBoostClassifier, CatBoostRegressor
        CATBOOST_AVAILABLE = True
except Exception:
    pass


class AutoMLTrainingEngine:
    """
    Modular production-grade AutoML Training Engine.
    Executes ThreadPoolExecutor safe parallel model training with per-model timeouts,
    failure sandboxing, robust feature sanitization, garbage collection,
    and a professional Balanced Evaluation Scoring framework.
    """
    def __init__(
        self,
        task_type: str, # binary, multiclass, regression
        timeout_seconds: float = 300.0,
        model_timeout_seconds: float = 60.0,
        is_cancelled: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[str, float, Dict[str, Any]], None]] = None
    ):
        self.task_type = task_type
        self.timeout_seconds = timeout_seconds
        self.model_timeout_seconds = model_timeout_seconds
        self.is_cancelled = is_cancelled
        self.progress_callback = progress_callback
        self.start_time = 0.0
        self.model_status = {}  # model_name -> status (completed, failed, timeout, skipped, incompatible dataset)
        self.model_errors = {}  # model_name -> error log/traceback
        self.lock = threading.Lock()
        
    def _check_safeguards(self):
        """
        Validates global timeouts and cancellation signals.
        """
        if self.is_cancelled and self.is_cancelled():
            raise KeyboardInterrupt("TRAINING_CANCELLED")
            
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError("TRAINING_TIMEOUT")

    def _validate_and_align_features(self, X: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
        """
        Validates feature columns of X against expected_features.
        Performs checks for duplicates, missing columns, and reorders to match training layout.
        Returns a cleanly aligned DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, "shape") and len(expected_features) == X.shape[1]:
                X = pd.DataFrame(X, columns=expected_features)
            else:
                raise AutoMLValidationError(
                    f"Expected inputs to be a pandas DataFrame, got {type(X)}."
                )

        dup_cols = X.columns[X.columns.duplicated()].tolist()
        if dup_cols:
            raise AutoMLValidationError(
                f"Defensive Validation Error: Input features contain duplicate columns: {dup_cols}"
            )

        missing_cols = [c for c in expected_features if c not in X.columns]
        if missing_cols:
            raise AutoMLValidationError(
                f"Defensive Validation Error: Input features are missing expected columns: {missing_cols}"
            )

        # Slice / Reorder strictly matching the expected column layout
        X_aligned = X[expected_features].copy()
        return X_aligned

    def get_available_models(self) -> List[str]:
        """
        Returns a list of supported models.
        """
        models = [
            "RandomForest",
            "ExtraTrees",
            "DecisionTree",
            "GradientBoosting",
            "KNN"
        ]
        if self.task_type in ["binary", "multiclass"]:
            models.append("LogisticRegression")
            models.append("SVM")
        else:
            models.append("LinearRegression")
            models.append("SVM")
            
        if XGB_AVAILABLE:
            models.append("XGBoost")
        if LGBM_AVAILABLE:
            models.append("LightGBM")
        if CATBOOST_AVAILABLE:
            models.append("CatBoost")
            
        return models

    def select_smart_models(self, X_shape: Tuple[int, int], manual_selection: Optional[List[str]] = None) -> List[str]:
        """
        Smart model selector. Limits memory usage and filters heavy models on massive datasets.
        """
        available = self.get_available_models()
        if manual_selection:
            return [m for m in manual_selection if m in available]
            
        selected = []
        rows, cols = X_shape
        total_cells = rows * cols
        
        # 1. Base estimators
        if self.task_type in ["binary", "multiclass"]:
            selected.append("LogisticRegression")
        else:
            selected.append("LinearRegression")
        selected.append("DecisionTree")
        selected.append("RandomForest")
        
        # Heavy Dataset Optimization Settings
        if total_cells > 1_000_000:
            # Giant dataset: Avoid SVM, CatBoost, standard Gradient Boosting (extremely slow)
            if LGBM_AVAILABLE:
                selected.append("LightGBM")
            if XGB_AVAILABLE:
                selected.append("XGBoost")
        elif total_cells > 200_000:
            # Medium-large dataset
            if LGBM_AVAILABLE:
                selected.append("LightGBM")
            if XGB_AVAILABLE:
                selected.append("XGBoost")
            selected.append("GradientBoosting")
            if rows < 10_000:
                selected.append("ExtraTrees")
        else:
            # Lightweight dataset
            selected.append("ExtraTrees")
            if XGB_AVAILABLE:
                selected.append("XGBoost")
            if LGBM_AVAILABLE:
                selected.append("LightGBM")
            if CATBOOST_AVAILABLE:
                selected.append("CatBoost")
            selected.append("GradientBoosting")
            selected.append("KNN")
            if rows < 3_000:
                selected.append("SVM")
                
        return selected

    def _instantiate_model(self, model_name: str) -> Any:
        is_cls = (self.task_type in ["binary", "multiclass"])
        
        if model_name == "LogisticRegression":
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "LinearRegression":
            return LinearRegression()
        elif model_name == "RandomForest":
            if is_cls:
                return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=1)
            else:
                return RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=1)
        elif model_name == "ExtraTrees":
            if is_cls:
                return ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=1)
            else:
                return ExtraTreesRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=1)
        elif model_name == "DecisionTree":
            if is_cls:
                return DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, random_state=42)
            else:
                return DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=42)
        elif model_name == "GradientBoosting":
            if is_cls:
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_name == "KNN":
            if is_cls:
                return KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            else:
                return KNeighborsRegressor(n_neighbors=5, n_jobs=1)
        elif model_name == "SVM":
            if is_cls:
                return SVC(probability=True, random_state=42)
            else:
                return SVR()
        elif model_name == "XGBoost" and XGB_AVAILABLE:
            if is_cls:
                return XGBClassifier(
                    n_estimators=100, max_depth=5, min_child_weight=4,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=1, verbosity=0
                )
            else:
                return XGBRegressor(
                    n_estimators=100, max_depth=5, min_child_weight=4,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=1, verbosity=0
                )
        elif model_name == "LightGBM" and LGBM_AVAILABLE:
            if is_cls:
                return LGBMClassifier(
                    n_estimators=100, max_depth=5, num_leaves=31, min_child_samples=10,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=1, verbose=-1
                )
            else:
                return LGBMRegressor(
                    n_estimators=100, max_depth=5, num_leaves=31, min_child_samples=10,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=1, verbose=-1
                )
        elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
            if is_cls:
                return CatBoostClassifier(iterations=100, random_state=42, verbose=0, thread_count=1)
            else:
                return CatBoostRegressor(iterations=100, random_state=42, verbose=0, thread_count=1)
                
        raise ValueError(f"Model '{model_name}' is not supported or not installed.")

    def evaluate_model_on_data(self, model: Any, X: Any, y: np.ndarray) -> Dict[str, float]:
        """
        Computes metric values for evaluation. Always aligns column orders dynamically.
        """
        if isinstance(X, pd.DataFrame):
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
                X = self._validate_and_align_features(X, expected_features)
                
        is_cls = (self.task_type in ["binary", "multiclass"])
        preds = model.predict(X)
        
        if is_cls:
            acc = float(accuracy_score(y, preds))
            prec = float(precision_score(y, preds, average="weighted", zero_division=0))
            rec = float(recall_score(y, preds, average="weighted", zero_division=0))
            f1 = float(f1_score(y, preds, average="weighted", zero_division=0))
            
            roc_auc = 0.5
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    if self.task_type == "binary":
                        roc_auc = float(roc_auc_score(y, probs[:, 1]))
                    else:
                        roc_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="weighted"))
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X)
                    roc_auc = float(roc_auc_score(y, scores, multi_class="ovr", average="weighted"))
            except Exception:
                pass
                
            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc
            }
        else:
            rmse = float(root_mean_squared_error(y, preds))
            mae = float(mean_absolute_error(y, preds))
            r2 = float(r2_score(y, preds))
            
            return {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

    # ------------------------------------------------
    # Balanced Composite Evaluation System
    # ------------------------------------------------
    def compute_balanced_composite(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        cv_scores: List[float],
        model_name: str
    ) -> Tuple[float, float, float, str, List[str]]:
        """
        Core SOL Balanced Scoring Engine.
        Analyzes train vs validation generalization gap, cross-validation stability, 
        and complexity to output the Composite Score and health statuses.
        """
        metric_key = "f1" if self.task_type in ["binary", "multiclass"] else "r2"
        
        # Base scores
        s_train = train_metrics.get(metric_key, 0.0)
        s_val = val_metrics.get(metric_key, 0.0)
        
        # Clip metrics to 0-1 bounds for scoring stability
        s_train_c = max(0.0, min(1.0, s_train))
        s_val_c = max(0.0, min(1.0, s_val))
        
        cv_mean = float(np.mean(cv_scores)) if cv_scores else s_val_c
        cv_std = float(np.std(cv_scores)) if cv_scores else 0.0
        
        cv_mean_c = max(0.0, min(1.0, cv_mean))
        
        # 1. Generalization Gap & Overfitting
        gen_gap = max(0.0, s_train_c - s_val_c)
        
        # Overfit penalties
        if gen_gap <= 0.05:
            overfit_penalty = 0.0
        elif gen_gap <= 0.15:
            overfit_penalty = 0.5 * (gen_gap - 0.05)
        else:
            overfit_penalty = 1.0 * (gen_gap - 0.05) + 0.10
            
        # 2. Perfect Score Warning (Overfitting / Leakage protection)
        # Models with validation score = 1.0 shouldn't automatically win
        perfect_score_penalty = 0.0
        if s_val == 1.0 and s_train == 1.0:
            if cv_mean < 0.95:
                # If CV is poor, massive target leakage / overfit warning
                perfect_score_penalty = 0.20
            else:
                perfect_score_penalty = 0.05
                
        # 3. Stability calculation
        stability = max(0.0, 1.0 - 5.0 * cv_std)
        
        # 4. Complexity Penalty
        # Complex algorithms (like deep trees, SVMs) have a minor penalty
        complexity_penalty = 0.0
        if model_name in ["RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]:
            complexity_penalty = 0.015
        elif model_name in ["SVM"]:
            complexity_penalty = 0.01
            
        # 5. Composite Score Formula
        composite_score = (
            0.4 * s_val_c + 
            0.4 * cv_mean_c + 
            0.2 * stability - 
            overfit_penalty - 
            perfect_score_penalty - 
            complexity_penalty
        )
        composite_score = float(max(0.0, min(1.0, composite_score)))
        
        # 6. Establish Health Categorizations & Labels
        labels = []
        
        # Overfit / Underfit classes
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
            
        # Variance
        if cv_std > 0.08:
            labels.append("High Variance")
            
        # Final Recommendation Tags
        if "Stable" in labels and cv_std < 0.04 and composite_score >= 0.70:
            labels.append("Recommended")
        else:
            labels.append("Experimental")
            
        return composite_score, gen_gap, cv_std, status_indicator, labels

    # ------------------------------------------------
    # Isolated Multi-threaded Worker Single Pipeline
    # ------------------------------------------------
    def _train_single_model_worker(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        cv_folds: int
    ) -> Optional[Dict[str, Any]]:
        """
        Isolated task thread function. Fits, cross-validates, and evaluates a single model.
        Runs full garbage collection to protect CPU resources.
        """
        t0 = time.time()
        try:
            # Instantiate model
            model = self._instantiate_model(name)
            
            # 1. Fit baseline model
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            
            # 2. Evaluate performance
            train_metrics = self.evaluate_model_on_data(model, X_train, y_train)
            val_metrics = self.evaluate_model_on_data(model, X_test, y_test)
            
            # 3. K-Fold Cross Validation
            cv_scores = []
            is_cls = (self.task_type in ["binary", "multiclass"])
            
            # Use StratifiedKFold for classification to maintain class balances
            cv_splitter = (
                StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                if is_cls else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            )
            
            metric_key = "f1" if is_cls else "r2"
            
            # Run CV splits
            for train_idx, val_idx in cv_splitter.split(X_train, y_train):
                # Slice matrices
                X_tr_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                
                cv_model = self._instantiate_model(name)
                cv_model.fit(X_tr_cv, y_tr_cv)
                
                cv_evals = self.evaluate_model_on_data(cv_model, X_val_cv, y_val_cv)
                cv_scores.append(cv_evals.get(metric_key, 0.0))
                
            # 4. Generate Balanced Composite Metrics & Labels
            composite_score, gen_gap, cv_std, status_indicator, labels = self.compute_balanced_composite(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                cv_scores=cv_scores,
                model_name=name
            )
            
            # Package output
            res = {
                "model_name": name,
                "fit_time": fit_time,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
                "cv_std": cv_std,
                "generalization_gap": gen_gap,
                "composite_score": composite_score,
                "status_indicator": status_indicator,
                "labels": labels,
                "model_instance": model
            }
            
            # Thread-safe status update
            with self.lock:
                self.model_status[name] = "completed"
                
            # Perform GC cleanup
            gc.collect()
            return res
            
        except Exception as e:
            import traceback
            error_msg = str(e) + "\n" + traceback.format_exc()
            
            status = "failed"
            if "timeout" in error_msg.lower():
                status = "timeout"
            elif "valueerror" in error_msg.lower() and "class" in error_msg.lower():
                status = "incompatible dataset"
                
            with self.lock:
                self.model_status[name] = status
                self.model_errors[name] = error_msg
                
            gc.collect()
            return None

    def train_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_names: List[str],
        cv_folds: int = 5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Phase 1: Safe Parallel Baseline Training using ThreadPoolExecutor.
        Each training worker is fully sandboxed. One failure NEVER stops execution.
        """
        self.start_time = time.time()
        
        if not isinstance(X_train, pd.DataFrame):
            raise AutoMLValidationError("X_train must be a pandas DataFrame to maintain feature schemas.")
            
        expected_features = list(X_train.columns)
        X_train = self._validate_and_align_features(X_train, expected_features)
        X_test = self._validate_and_align_features(X_test, expected_features)
        
        # Heavy Dataset Optimization Check
        total_cells = X_train.shape[0] * X_train.shape[1]
        if total_cells > 200_000:
            # Automatically scale down CV folds to preserve memory/runtimes
            cv_folds = min(3, cv_folds)
            
        leaderboard = []
        trained_instances = {}
        
        total_models = len(model_names)
        
        with self.lock:
            for name in model_names:
                self.model_status[name] = "pending"
                
        # Fire concurrent threads
        # We cap worker count to CPU count to prevent excessive OS scheduling thrashing
        max_workers = min(4, os.cpu_count() or 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {}
            for name in model_names:
                with self.lock:
                    self.model_status[name] = "running"
                if self.progress_callback:
                    self.progress_callback(name, 0.0, {"status": "training"})
                    
                # Submit worker thread with isolated context copies
                future = executor.submit(
                    self._train_single_model_worker,
                    name,
                    X_train.copy(),
                    y_train.copy(),
                    X_test.copy(),
                    y_test.copy(),
                    cv_folds
                )
                future_to_model[future] = name
                
            # Process completions as they finish
            completed_count = 0
            for future in as_completed(future_to_model):
                name = future_to_model[future]
                completed_count += 1
                progress_pct = completed_count / total_models
                
                try:
                    # Fetch outcome with explicit safety timeout limits
                    res = future.result(timeout=self.model_timeout_seconds)
                    if res is not None:
                        leaderboard.append(res)
                        trained_instances[name] = res.pop("model_instance")
                        
                        if self.progress_callback:
                            self.progress_callback(name, progress_pct, {"status": "success", "metrics": res})
                    else:
                        if self.progress_callback:
                            self.progress_callback(name, progress_pct, {"status": self.model_status.get(name, "failed"), "error": self.model_errors.get(name, "Unknown failure.")})
                except Exception as t_err:
                    import traceback
                    err_msg = f"Thread timeout/error for {name}: {t_err}\n{traceback.format_exc()}"
                    with self.lock:
                        self.model_status[name] = "timeout" if "timeout" in str(t_err).lower() else "failed"
                        self.model_errors[name] = err_msg
                    if self.progress_callback:
                        self.progress_callback(name, progress_pct, {"status": "timeout" if "timeout" in str(t_err).lower() else "failed", "error": str(t_err)})
                        
        # Final failed logs serialization to disk
        self.serialize_failed_models_log()
        
        # Sort leaderboard using our intelligent composite score (descending)
        if leaderboard:
            leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)
            
        return leaderboard, trained_instances

    def serialize_failed_models_log(self):
        """
        Saves all thread-captured failures to logs/failed_models.json
        """
        os.makedirs("logs", exist_ok=True)
        failed_log = {
            m: self.model_errors.get(m, "No traceback captured.")
            for m, st in self.model_status.items() if st in ["failed", "timeout", "incompatible dataset"]
        }
        with open(os.path.join("logs", "failed_models.json"), "w", encoding="utf-8") as f:
            json.dump(failed_log, f, indent=4)

    def deep_optimize_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        best_model_name: str,
        best_model: Any,
        cv_folds: int = 5
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Phase 2: Deep Optimization.
        Performs quick hyperparameter tuning via RandomizedSearchCV and returns
        the tuned model with complete balanced composite diagnostics.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise AutoMLValidationError("X_train must be a pandas DataFrame.")
            
        expected_features = list(X_train.columns)
        X_train = self._validate_and_align_features(X_train, expected_features)
        X_test = self._validate_and_align_features(X_test, expected_features)
        
        is_cls = (self.task_type in ["binary", "multiclass"])
        
        # Search grids
        param_dists = {}
        if best_model_name == "RandomForest":
            param_dists = {
                "n_estimators": [50, 100, 150],
                "max_depth": [6, 10, 15],
                "min_samples_leaf": [2, 4, 8],
                "min_samples_split": [5, 10]
            }
        elif best_model_name == "ExtraTrees":
            param_dists = {
                "n_estimators": [50, 100, 150],
                "max_depth": [6, 10, 15],
                "min_samples_leaf": [2, 4, 8],
                "min_samples_split": [5, 10]
            }
        elif best_model_name == "DecisionTree":
            param_dists = {
                "max_depth": [4, 6, 8, 10],
                "min_samples_leaf": [2, 4, 8],
                "min_samples_split": [5, 10]
            }
        elif best_model_name == "LogisticRegression":
            param_dists = {
                "C": [0.01, 0.1, 1.0, 10.0]
            }
        elif best_model_name == "SVM":
            param_dists = {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"]
            }
        elif best_model_name == "GradientBoosting":
            param_dists = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5]
            }
        elif best_model_name == "XGBoost" and XGB_AVAILABLE:
            param_dists = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6]
            }
        elif best_model_name == "LightGBM" and LGBM_AVAILABLE:
            param_dists = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [15, 31]
            }
        elif best_model_name == "CatBoost" and CATBOOST_AVAILABLE:
            param_dists = {
                "iterations": [50, 100],
                "learning_rate": [0.01, 0.1]
            }
            
        if not param_dists:
            # Fallback
            train_m = self.evaluate_model_on_data(best_model, X_train, y_train)
            val_m = self.evaluate_model_on_data(best_model, X_test, y_test)
            return best_model, {**val_m, "train_metrics": train_m, "val_metrics": val_m, "composite_score": val_m.get("f1" if is_cls else "r2", 0.5), "status_indicator": "Stable", "labels": ["Stable"]}
            
        from sklearn.model_selection import RandomizedSearchCV
        scoring = "f1_weighted" if is_cls else "neg_root_mean_squared_error"
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_cls else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            estimator=self._instantiate_model(best_model_name),
            param_distributions=param_dists,
            n_iter=5,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=1
        )
        
        try:
            self._check_safeguards()
            if self.progress_callback:
                self.progress_callback(best_model_name, 0.3, {"status": "tuning"})
            search.fit(X_train, y_train)
            tuned_model = search.best_estimator_
            
            # Recalculate metrics
            train_metrics = self.evaluate_model_on_data(tuned_model, X_train, y_train)
            val_metrics = self.evaluate_model_on_data(tuned_model, X_test, y_test)
            
            # CV score check
            cv_scores = []
            for tr_idx, vl_idx in cv.split(X_train, y_train):
                X_tr_cv, X_val_cv = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
                y_tr_cv, y_val_cv = y_train[tr_idx], y_train[vl_idx]
                cv_model = self._instantiate_model(best_model_name)
                cv_model.set_params(**search.best_params_)
                cv_model.fit(X_tr_cv, y_tr_cv)
                cv_evals = self.evaluate_model_on_data(cv_model, X_val_cv, y_val_cv)
                cv_scores.append(cv_evals.get("f1" if is_cls else "r2", 0.0))
                
            composite_score, gen_gap, cv_std, status_indicator, labels = self.compute_balanced_composite(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                cv_scores=cv_scores,
                model_name=best_model_name
            )
            
            res = {
                "model_name": best_model_name,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
                "cv_std": cv_std,
                "generalization_gap": gen_gap,
                "composite_score": composite_score,
                "status_indicator": status_indicator,
                "labels": labels,
                # Copy baseline metrics as fallback structure keys
                "f1": val_metrics.get("f1", 0.0) if is_cls else 0.0,
                "accuracy": val_metrics.get("accuracy", 0.0) if is_cls else 0.0,
                "precision": val_metrics.get("precision", 0.0) if is_cls else 0.0,
                "recall": val_metrics.get("recall", 0.0) if is_cls else 0.0,
                "roc_auc": val_metrics.get("roc_auc", 0.0) if is_cls else 0.0,
                "rmse": val_metrics.get("rmse", 0.0) if not is_cls else 0.0,
                "mae": val_metrics.get("mae", 0.0) if not is_cls else 0.0,
                "r2": val_metrics.get("r2", 0.0) if not is_cls else 0.0,
            }
            
            if self.progress_callback:
                self.progress_callback(best_model_name, 1.0, {"status": "tuned", "metrics": res})
            return tuned_model, res
        except Exception:
            # Fallback
            train_m = self.evaluate_model_on_data(best_model, X_train, y_train)
            val_m = self.evaluate_model_on_data(best_model, X_test, y_test)
            return best_model, {**val_m, "train_metrics": train_m, "val_metrics": val_m, "composite_score": val_m.get("f1" if is_cls else "r2", 0.5), "status_indicator": "Stable", "labels": ["Stable"]}

    def extract_feature_importance(self, model: Any, feature_names: List[str], X_val: Any, y_val: Any) -> List[Dict[str, Any]]:
        """
        Extracts feature importance with fallbacks. Align inputs strictly before running.
        """
        if isinstance(X_val, pd.DataFrame):
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
                X_val = self._validate_and_align_features(X_val, expected_features)
                
        importances = None
        
        # 1. Tree native importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            
        # 2. Linear coef
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            total = np.sum(importances)
            if total > 0:
                importances = importances / total
                
        # 3. Permutation importance fallback
        if importances is None:
            try:
                sample_size = min(len(X_val), 300)
                indices = np.random.choice(len(X_val), sample_size, replace=False)
                
                if isinstance(X_val, pd.DataFrame):
                    X_sample = X_val.iloc[indices]
                else:
                    X_sample = X_val[indices]
                    
                if isinstance(y_val, (pd.Series, pd.DataFrame, np.ndarray)):
                    if hasattr(y_val, "iloc"):
                        y_sample = y_val.iloc[indices]
                    else:
                        y_sample = y_val[indices]
                else:
                    y_sample = np.array(y_val)[indices]
                    
                res = permutation_importance(model, X_sample, y_sample, n_repeats=2, random_state=42, n_jobs=1)
                importances = np.abs(res.importances_mean)
                total = np.sum(importances)
                if total > 0:
                    importances = importances / total
            except Exception:
                pass
                
        if importances is None or len(importances) != len(feature_names):
            importances = np.zeros(len(feature_names))
            
        result = [
            {"feature": f, "importance": float(i)}
            for f, i in zip(feature_names, importances)
        ]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result
