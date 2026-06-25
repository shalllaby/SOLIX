import os
import sys
import json
import time
import shutil
import tempfile
import threading
import logging
from pathlib import Path

# Setup Terminal Logger with SOL format
logger = logging.getLogger("SOL.KaggleOrchestrator")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Auto-install/import Kaggle API extended wrapper
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


def _update_db_job_status(task_id: str, status: str, error_msg: str = None, stats: dict = None, accuracy: float = None):
    try:
        from backend.database import SessionLocal
        from backend.models import JobRecord
        db = SessionLocal()
        try:
            job = db.query(JobRecord).filter(JobRecord.task_id == task_id).first()
            if job:
                job.status = status
                if error_msg:
                    job.error_message = error_msg
                if accuracy is not None:
                    job.accuracy_rate = accuracy
                elif stats and "accuracy" in stats:
                    job.accuracy_rate = stats["accuracy"]
                
                if stats:
                    if "rows_after" in stats:
                        job.row_count = stats["rows_after"]
                    elif "rows" in stats:
                        job.row_count = stats["rows"]
                    if "cols_after" in stats:
                        job.col_count = stats["cols_after"]
                db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error("Failed to update JobRecord in DB: %s", e)


def execute_local_fallback(
    task_id: str,
    dataset_id: str,
    strategy_level: str,
    goal: str | None,
    approved_actions: str | None,
    error_msg: str
):
    """
    Executes the standard cleaning workflow locally when Kaggle runs fail or time out.
    Ensures seamless service continuity without disrupting the UI/UX.
    """
    import pandas as pd
    import numpy as np
    
    # Import core engines and main helper functions inside function context to prevent circular imports
    from core.cleaner import SmartDataCleaner
    from backend.tools.audit.engine import AuditReportBuilder
    from backend.tools.viz_engine.engine import VizEngine
    from backend.main import _build_diff_map, _build_nan_map, _df_to_records, _build_strategy, analyzer
    from backend.store import (
        _store, _store_parquet_path, _store_tasks, _store_filename, _store_goals, _audit_store, _viz_store
    )
    
    logger.warning("Kaggle execution failed: '%s'. Triggering local fallback cleaner...", error_msg)
    
    # Track fallback warning inside the task object for audit purposes
    if task_id in _store_tasks:
        _store_tasks[task_id]["warnings"] = _store_tasks[task_id].get("warnings", []) + [
            f"Kaggle execution failed: {error_msg}. Fell back to local cleaning."
        ]
        
    try:
        parquet_path = _store_parquet_path.get(dataset_id)
        if not parquet_path:
            raise ValueError(f"Dataset Parquet path not registered for dataset_id: {dataset_id}")
            
        raw_df = pd.read_parquet(parquet_path)
        
        # Build metadata for local strategist
        try:
            metadata = analyzer.analyze_file(filename=_store_filename.get(dataset_id, "data.parquet"), file_path=parquet_path)
        except Exception:
            metadata = {"columns_info": []}
            
        strategy_json = _build_strategy(strategy_level, metadata, goal)
        
        policy_config = None
        if approved_actions:
            try:
                approved_list = json.loads(approved_actions)
                if isinstance(approved_list, list):
                    policy_config = {"approved_actions": approved_list}
            except Exception as e:
                logger.error("Failed to parse approved_actions configuration: %s", e)
                
        # Run Local Smart Cleaner
        cleaner = SmartDataCleaner(raw_df, policy_config=policy_config)
        result = cleaner.execute_strategy(strategy_json)
        cleaned_df, report = result if isinstance(result, tuple) else (result, {"actions": []})
        
        # Save output
        cleaned_parquet_path = parquet_path.replace(".parquet", "_cleaned.parquet")
        cleaned_df.to_parquet(cleaned_parquet_path, index=False)
        
        _store_parquet_path[dataset_id + "_prev"] = parquet_path
        _store_parquet_path[dataset_id] = cleaned_parquet_path
        
        cleaned_id = dataset_id + "_cleaned"
        if len(cleaned_df) <= 50000:
            _store[cleaned_id] = cleaned_df
        else:
            _store[cleaned_id] = cleaned_df.head(500)
            
        # Preview records (up to 500 rows)
        raw_preview = raw_df.head(500)
        cleaned_preview = cleaned_df.head(500)
        diff_map = _build_diff_map(raw_preview, cleaned_preview)
        nan_map = _build_nan_map(raw_preview)
        
        missing_before = int(raw_df.isna().sum().sum())
        missing_after = int(cleaned_df.isna().sum().sum())
        
        # Local Audit log
        audit_builder = AuditReportBuilder(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            cleaner_report=report,
            strategy_used=strategy_level,
            filename=_store_filename.get(dataset_id, "dataset.csv"),
            user_goal=_store_goals.get(dataset_id, goal),
            dataset_id=dataset_id,
            strategy_json=strategy_json,
        )
        audit_log = audit_builder.build()
        _audit_store[dataset_id] = audit_log
        audit_id = audit_log["audit_id"]
        
        # Local Viz payload
        try:
            viz_cmp = VizEngine(raw_df=raw_df.head(10000), cleaned_df=cleaned_df.head(10000))
            _viz_store[dataset_id] = viz_cmp.comparison()
        except Exception as _viz_err:
            logger.warning("VizEngine comparison error: %s", _viz_err)
            
        # Finalize status mapping
        _store_tasks[task_id]["progress"] = 100
        _store_tasks[task_id]["status"] = "completed"
        _store_tasks[task_id]["result"] = {
            "dataset_id": dataset_id,
            "cleaned_dataset_id": cleaned_id,
            "strategy_used": strategy_level,
            "audit_id": audit_id,
            "audit_log": audit_log,
            "report": {
                **report,
                "cleaning_strategy": strategy_json.get("cleaning_strategy", {}),
            },
            "stats": {
                "rows_before": len(raw_df),
                "rows_after": len(cleaned_df),
                "missing_before": missing_before,
                "missing_after": missing_after,
                "cells_fixed": missing_before - missing_after,
            },
            "raw_data": _df_to_records(raw_preview),
            "cleaned_data": _df_to_records(cleaned_preview),
            "diff_map": diff_map,
            "nan_map": nan_map,
        }
        logger.info("Local fallback completed successfully for task '%s'.", task_id)
        _update_db_job_status(task_id, "completed", stats={
            "rows_after": len(cleaned_df),
            "cols_after": len(cleaned_df.columns),
            "accuracy": audit_log.get("truth_confidence_score", 100.0) if isinstance(audit_log, dict) else 100.0
        })
        
    except Exception as local_err:
        logger.error("Critical Failure: Local fallback cleaning failed: %s", local_err)
        if task_id in _store_tasks:
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = f"Local fallback failure: {str(local_err)}"
            _update_db_job_status(task_id, "failed", error_msg=f"Local fallback failure: {str(local_err)}")


def retry_kaggle_call(func, *args, max_retries=3, initial_delay=2, backoff_factor=2, **kwargs):
    """
    Executes a Kaggle API function with retry logic and exponential backoff.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error("Kaggle API call %s failed after %d attempts: %s", func.__name__, max_retries, e)
                raise e
            
            logger.warning(
                "Kaggle API call %s failed (attempt %d/%d) due to network/timeout. Retrying in %ds... Error: %s",
                func.__name__, attempt, max_retries, delay, e
            )
            time.sleep(delay)
            delay *= backoff_factor

class KaggleStudioOrchestrator:
    """
    Manages the lifecycle of the remote Kaggle Data Cleaning Studio jobs:
    authenticates credentials, pushes raw datasets and strategy scripts,
    polls progress, and unifies local fallbacks.
    """
    def __init__(self, username: str = None, api_token: str = None):
        self.username = username or os.environ.get("KAGGLE_USERNAME", "al_dalil_governance_service")
        self.api_token = api_token or os.environ.get("KAGGLE_API_TOKEN", "KGAT_0034d6fd413ada3d3b57d06d1736d0ae")
        self.api = None
        self.is_running = False
        
        if KAGGLE_AVAILABLE:
            try:
                self.api = self._authenticate()
            except Exception as e:
                logger.error("Kaggle Auth Initialization failed: %s. Local fallback will be enforced.", e)

    def _authenticate(self) -> "KaggleApi":
        """Authenticates with the Kaggle API using custom token."""
        # Store original environment variables to restore them later
        orig_user = os.environ.get("KAGGLE_USERNAME")
        orig_key = os.environ.get("KAGGLE_KEY")

        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        legacy_json = kaggle_dir / "kaggle.json"
        if legacy_json.exists():
            try:
                legacy_json.unlink()
            except Exception as e:
                logger.warning("Could not delete legacy kaggle.json: %s", str(e))
                
        access_token_path = kaggle_dir / "access_token"
        with open(access_token_path, "w", encoding="utf-8") as f:
            f.write(self.api_token.strip())
            
        try:
            os.chmod(access_token_path, 0o600)
        except Exception:
            pass
            
        os.environ["KAGGLE_API_TOKEN"] = self.api_token.strip()
        
        api = KaggleApi()
        api.authenticate()

        # Restore original values so subsequent requests/checks still see them
        if orig_user is not None:
            os.environ["KAGGLE_USERNAME"] = orig_user
        if orig_key is not None:
            os.environ["KAGGLE_KEY"] = orig_key

        # Inject connection and read timeout values onto ApiClient Configuration
        try:
            config = getattr(api, "config", None) or getattr(api, "configuration", None)
            if not config and hasattr(api, "api_client") and api.api_client:
                config = getattr(api.api_client, "configuration", None)
                
            if config:
                config.connection_timeout = 60
                config.read_timeout = 60
                
                # Support standard proxy environment vars
                http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
                https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
                if https_proxy:
                    config.proxy = https_proxy
                elif http_proxy:
                    config.proxy = http_proxy
                logger.info("Kaggle API client timeout and proxy configuration applied.")
        except Exception as config_err:
            logger.warning("Could not set custom Kaggle API config timeout/proxy: %s", config_err)

        return api

    def run_remote_clean(
        self,
        task_id: str,
        dataset_id: str,
        strategy_level: str,
        goal: str | None,
        approved_actions: str | None,
    ):
        """
        Prepares and pushes the execution package to Kaggle, triggers the remote kernel, 
        and kicks off a monitoring polling thread.
        """
        from backend.store import _store_tasks, _store_parquet_path, _store_filename, _store_goals
        from backend.utils.bundler import create_studio_bundle
        
        if not self.api:
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = "Kaggle API token authentication is not configured or failed."
            _update_db_job_status(task_id, "failed", error_msg="Kaggle API token authentication is not configured or failed.")
            return
            
        _store_tasks[task_id]["status"] = "processing"
        _store_tasks[task_id]["progress"] = 10
        self.is_running = True
        
        temp_dataset_dir = None
        temp_kernel_dir = None
        
        try:
            parquet_path = _store_parquet_path.get(dataset_id)
            if not parquet_path:
                raise ValueError("Source dataset file path not registered.")
                
            # To solve the Case Mismatch issue safely, use the lowercase version of the initialized username
            safe_username = self.username.lower() if self.username else ""
            
            # Kaggle slugs must be lowercase, alphanumeric, 5-50 characters
            clean_uuid = "".join(c for c in task_id if c.isalnum()).lower()
            dataset_slug = f"solcleanin{clean_uuid}"
            kernel_slug = f"solcleanker{clean_uuid}"
            full_dataset_ref = f"{safe_username}/{dataset_slug}"
            full_kernel_ref = f"{safe_username}/{kernel_slug}"
            
            # 1. Create temporary directory for dataset packaging
            temp_dataset_dir = Path(tempfile.mkdtemp())
            logger.info("Packaging dataset elements in: '%s'...", temp_dataset_dir)
            
            # Copy Parquet dataset
            shutil.copy(parquet_path, temp_dataset_dir / "raw.parquet")
            
            # Generate source code zip bundle
            zip_out = temp_dataset_dir / "studio_core.zip"
            create_studio_bundle(str(zip_out))
            
            # Build strategy parameters
            from backend.main import _build_strategy, analyzer
            try:
                metadata = analyzer.analyze_file(filename=_store_filename.get(dataset_id, "data.parquet"), file_path=parquet_path)
            except Exception:
                metadata = {"columns_info": []}
                
            strategy_json = _build_strategy(strategy_level, metadata, goal)
            
            policy_config = None
            if approved_actions:
                try:
                    approved_list = json.loads(approved_actions)
                    if isinstance(approved_list, list):
                        policy_config = {"approved_actions": approved_list}
                except Exception as e:
                    logger.warning("Parsing warning on approved_actions: %s", e)
            
            # Write strategy configurations (securely inject local GROQ_API_KEY)
            config_payload = {
                "dataset_id": dataset_id,
                "filename": _store_filename.get(dataset_id, "dataset.csv"),
                "user_goal": _store_goals.get(dataset_id, goal or ""),
                "strategy_level": strategy_level,
                "strategy_json": strategy_json,
                "policy_config": policy_config,
                "groq_api_key": os.environ.get("GROQ_API_KEY", "")
            }
            
            with open(temp_dataset_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=4)
                
            # Dataset metadata file for Kaggle
            meta = {
                "title": dataset_slug,
                "id": full_dataset_ref,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(temp_dataset_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
                
            _store_tasks[task_id]["progress"] = 25
            
            # Push payload to Kaggle
            logger.info("Uploading private payload dataset to Kaggle: '%s'...", full_dataset_ref)
            retry_kaggle_call(
                self.api.dataset_create_new,
                folder=str(temp_dataset_dir),
                public=False,
                quiet=True
            )
            
            # Wait for dataset compilation on Kaggle
            logger.info("Waiting for dataset mount readiness on remote servers...")
            max_wait_seconds = 180
            poll_interval = 2
            elapsed = 0
            dataset_ready = False
            
            while elapsed < max_wait_seconds:
                try:
                    status_data = retry_kaggle_call(
                        self.api.dataset_status,
                        full_dataset_ref
                    )
                    status_str = str(getattr(status_data, "status", status_data)).lower()
                    if "ready" in status_str:
                        dataset_ready = True
                        break
                except Exception as err:
                    logger.warning("Dataset polling check failed: %s (Will retry)", err)
                time.sleep(poll_interval)
                elapsed += poll_interval
                
            if not dataset_ready:
                raise TimeoutError("Kaggle dataset processing timed out (180 seconds limit reached).")
                
            _store_tasks[task_id]["progress"] = 45
            
            # 2. Package Kernel files
            temp_kernel_dir = Path(tempfile.mkdtemp())
            logger.info("Packaging kernel elements in: '%s'...", temp_kernel_dir)
            
            # Read template script from project root, inject credentials & dataset ref
            root_kernel_path = Path(__file__).resolve().parent.parent.parent / "kaggle_cleaning_kernel.py"
            with open(root_kernel_path, "r", encoding="utf-8") as f:
                kernel_code = f.read()
                
            # Replace placeholder variables
            kernel_code = kernel_code.replace('__KAGGLE_USERNAME_PLACEHOLDER__', self.username)
            kernel_code = kernel_code.replace('__KAGGLE_KEY_PLACEHOLDER__', self.api_token)
            kernel_code = kernel_code.replace('__GROQ_API_KEY_PLACEHOLDER__', os.environ.get("GROQ_API_KEY", ""))
            kernel_code = kernel_code.replace('__DATASET_REF_PLACEHOLDER__', full_dataset_ref)
            
            with open(temp_kernel_dir / "kaggle_cleaning_kernel.py", "w", encoding="utf-8") as f:
                f.write(kernel_code)
            
            # Kernel metadata file for Kaggle (mounts the dataset)
            kernel_meta = {
                "id": full_kernel_ref,
                "title": kernel_slug,
                "code_file": "kaggle_cleaning_kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [full_dataset_ref],
                "dataset_data_sources": [full_dataset_ref],
                "competition_sources": [],
                "kernel_sources": []
            }
            
            with open(temp_kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
                json.dump(kernel_meta, f, indent=4)
                
            logger.info("Pushing remote Kaggle execution kernel: '%s'...", full_kernel_ref)
            retry_kaggle_call(
                self.api.kernels_push,
                str(temp_kernel_dir)
            )
            
            # Construct live Kaggle Kernel URL
            kernel_url = f"https://www.kaggle.com/code/{safe_username}/{kernel_slug}"
            _store_tasks[task_id]["kernel_url"] = kernel_url
            
            _store_tasks[task_id]["progress"] = 60
            
            # Run asynchronous polling thread
            threading.Thread(
                target=self._poll_kernel_status,
                args=(task_id, dataset_id, full_kernel_ref, full_dataset_ref, strategy_level, goal, approved_actions),
                daemon=True
            ).start()
            
        except Exception as e:
            self.is_running = False
            logger.error("Failed to trigger remote cleaning on Kaggle: %s", e)
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = f"Submission error: {str(e)}"
            _update_db_job_status(task_id, "failed", error_msg=f"Submission error: {str(e)}")
            
        finally:
            # Clean up local staging folders
            if temp_dataset_dir and temp_dataset_dir.exists():
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)
            if temp_kernel_dir and temp_kernel_dir.exists():
                shutil.rmtree(temp_kernel_dir, ignore_errors=True)

    def _poll_kernel_status(
        self,
        task_id: str,
        dataset_id: str,
        kernel_ref: str,
        dataset_ref: str,
        strategy_level: str,
        goal: str | None,
        approved_actions: str | None
    ):
        """Background thread worker to monitor Kaggle kernel execution progress."""
        from backend.store import _store_tasks
        
        try:
            max_poll_seconds = 86400  # 24 hours execution limit
            poll_interval = 3
            elapsed = 0
            
            logger.info("Remote monitoring thread active for kernel '%s'...", kernel_ref)
            while elapsed < max_poll_seconds:
                status = self._get_kernel_status(kernel_ref)
                
                if status == "complete":
                    self._download_and_inject(task_id, dataset_id, kernel_ref, dataset_ref, strategy_level, goal)
                    return
                elif status == "error":
                    _store_tasks[task_id]["status"] = "failed"
                    _store_tasks[task_id]["error"] = "Kaggle execution failed on remote server."
                    _update_db_job_status(task_id, "failed", error_msg="Kaggle execution failed on remote server.")
                    self._safe_cleanup_remote(dataset_ref)
                    return
                    
                # Scale task progress metrics gradually
                progress_offset = min(60 + int((elapsed / max_poll_seconds) * 30), 90)
                if task_id in _store_tasks:
                    _store_tasks[task_id]["progress"] = progress_offset
                    
                time.sleep(poll_interval)
                elapsed += poll_interval
                
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = "Kaggle kernel run timed out."
            _update_db_job_status(task_id, "failed", error_msg="Kaggle kernel run timed out.")
            self._safe_cleanup_remote(dataset_ref)
        finally:
            self.is_running = False

    def _get_kernel_status(self, kernel_ref: str) -> str:
        """Query kernel status state from Kaggle."""
        try:
            status_data = retry_kaggle_call(
                self.api.kernels_status,
                kernel_ref
            )
            status_val = getattr(status_data, "status", status_data)
            status_str = str(getattr(status_val, "value", getattr(status_val, "name", status_val))).lower().strip()
            
            if "complete" in status_str or status_str in ["2", "3"]:
                return "complete"
            elif status_str in ("4", "error", "failed"):
                return "error"
            elif status_str in ("1", "running"):
                return "running"
            elif status_str in ("0", "queued"):
                return "queued"
            return status_str
        except Exception as e:
            logger.error("Error querying remote status for kernel '%s': %s", kernel_ref, e)
            return "error"

    def _download_and_inject(
        self,
        task_id: str,
        dataset_id: str,
        kernel_ref: str,
        dataset_ref: str,
        strategy_level: str,
        goal: str | None
    ):
        """Downloads, unpacks, and maps remote cleaning artifacts to the local store systems."""
        import pandas as pd
        from backend.store import (
            _store, _store_parquet_path, _store_tasks, _store_filename, _store_goals, _audit_store, _viz_store
        )
        from backend.main import _df_to_records, _build_diff_map, _build_nan_map, _build_strategy, analyzer
        
        logger.info("Remote job successful. Downloading outputs from Kaggle...")
        
        temp_out_dir = Path("temp_snapshots") / f"out_{task_id}"
        temp_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            retry_kaggle_call(
                self.api.kernels_output,
                kernel_ref,
                path=str(temp_out_dir),
                force=True
            )
            
            cleaned_parquet_path = temp_out_dir / "cleaned.parquet"
            audit_json_path = temp_out_dir / "audit.json"
            viz_payload_path = temp_out_dir / "viz_payload.json"
            
            if not cleaned_parquet_path.exists():
                raise FileNotFoundError("Outputs folder is missing 'cleaned.parquet'.")
                
            parquet_path = _store_parquet_path.get(dataset_id)
            if not parquet_path:
                parquet_path = str(Path("temp_snapshots") / f"{dataset_id}.parquet")
                
            final_cleaned_path = parquet_path.replace(".parquet", "_cleaned.parquet")
            shutil.move(str(cleaned_parquet_path), final_cleaned_path)
            
            import concurrent.futures
            
            def load_cleaned():
                cleaned_df = pd.read_parquet(final_cleaned_path)
                cleaned_id = dataset_id + "_cleaned"
                if len(cleaned_df) <= 50000:
                    _store[cleaned_id] = cleaned_df
                else:
                    _store[cleaned_id] = cleaned_df.head(500)
                _store_parquet_path[dataset_id + "_prev"] = parquet_path
                _store_parquet_path[dataset_id] = final_cleaned_path
                return cleaned_df

            def load_audit():
                if audit_json_path.exists():
                    with open(audit_json_path, "r", encoding="utf-8") as f:
                        audit_log = json.load(f)
                else:
                    audit_log = {"error": "Remote audit payload not generated."}
                _audit_store[dataset_id] = audit_log
                return audit_log

            def load_viz():
                if viz_payload_path.exists():
                    with open(viz_payload_path, "r", encoding="utf-8") as f:
                        viz_payload = json.load(f)
                else:
                    viz_payload = {"error": "Remote viz charts payload not generated."}
                _viz_store[dataset_id] = viz_payload
                return viz_payload

            def load_raw():
                return pd.read_parquet(parquet_path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_cleaned = executor.submit(load_cleaned)
                future_audit = executor.submit(load_audit)
                future_viz = executor.submit(load_viz)
                future_raw = executor.submit(load_raw)
                
                cleaned_df = future_cleaned.result()
                audit_log = future_audit.result()
                viz_payload = future_viz.result()
                raw_df = future_raw.result()
            
            cleaned_id = dataset_id + "_cleaned"
            audit_id = audit_log.get("audit_id") if isinstance(audit_log, dict) else None
            
            # Generate preview layouts
            raw_preview = raw_df.head(500)
            cleaned_preview = cleaned_df.head(500)
            diff_map = _build_diff_map(raw_preview, cleaned_preview)
            nan_map = _build_nan_map(raw_preview)
            
            missing_before = int(raw_df.isna().sum().sum())
            missing_after = int(cleaned_df.isna().sum().sum())
            
            try:
                metadata = analyzer.analyze_file(filename=_store_filename.get(dataset_id, "data.parquet"), file_path=parquet_path)
            except Exception:
                metadata = {"columns_info": []}
            strategy_json = _build_strategy(strategy_level, metadata, goal)
            
            # Populate task details
            _store_tasks[task_id]["progress"] = 100
            _store_tasks[task_id]["status"] = "completed"
            _store_tasks[task_id]["result"] = {
                "dataset_id": dataset_id,
                "cleaned_dataset_id": cleaned_id,
                "strategy_used": strategy_level,
                "audit_id": audit_id,
                "audit_log": audit_log,
                "report": {
                    "actions": audit_log.get("actions_log", []),
                    "cleaning_strategy": strategy_json.get("cleaning_strategy", {}),
                    "truth_confidence_score": audit_log.get("truth_confidence_score", 100.0)
                },
                "stats": {
                    "rows_before": len(raw_df),
                    "rows_after": len(cleaned_df),
                    "missing_before": missing_before,
                    "missing_after": missing_after,
                    "cells_fixed": missing_before - missing_after,
                },
                "raw_data": _df_to_records(raw_preview),
                "cleaned_data": _df_to_records(cleaned_preview),
                "diff_map": diff_map,
                "nan_map": nan_map,
            }
            logger.info("Successfully loaded and injected Kaggle output artifacts for task '%s'.", task_id)
            stats = {
                "rows_after": len(cleaned_df),
                "cols_after": len(cleaned_df.columns),
                "accuracy": audit_log.get("truth_confidence_score", 100.0) if isinstance(audit_log, dict) else 100.0
            }
            _update_db_job_status(task_id, "completed", stats=stats)
            
        except Exception as e:
            logger.error("Error unpacking downloaded artifacts: %s", e)
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = f"Extraction failure: {str(e)}"
            _update_db_job_status(task_id, "failed", error_msg=f"Extraction failure: {str(e)}")
            
        finally:
            shutil.rmtree(temp_out_dir, ignore_errors=True)
            self._safe_cleanup_remote(dataset_ref)

    def _safe_cleanup_remote(self, dataset_ref: str):
        """Securely deletes the input payload dataset on Kaggle to prevent storage threshold leakage."""
        try:
            logger.info("Requesting Kaggle deletion for temp dataset: '%s'...", dataset_ref)
            parts = dataset_ref.split('/')
            safe_username = self.username.lower() if self.username else ""
            if len(parts) == 2:
                owner_slug, dataset_slug = parts[0], parts[1]
            else:
                owner_slug, dataset_slug = safe_username, dataset_ref
            import builtins
            original_input = builtins.input
            try:
                builtins.input = lambda *args, **kwargs: 'yes'
                retry_kaggle_call(
                    self.api.dataset_delete,
                    owner_slug,
                    dataset_slug
                )
            finally:
                builtins.input = original_input
            logger.info("[SUCCESS] Kaggle temp dataset deleted successfully.")
        except Exception as e:
            logger.warning("Could not execute remote cleanup on dataset '%s': %s", dataset_ref, e)
