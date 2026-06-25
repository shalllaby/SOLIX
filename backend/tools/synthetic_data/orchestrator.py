import os
import sys
import builtins
import locale

# Global monkeypatch to force UTF-8 encoding on Windows, avoiding charmap encoding errors
locale.getpreferredencoding = lambda *args: 'utf-8'

original_open = builtins.open
def custom_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if 'b' not in mode and encoding is None:
        encoding = 'utf-8'
    return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
builtins.open = custom_open

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass

import json
import time
import shutil
import tempfile
import threading
import logging
import numpy as np
from pathlib import Path

# Setup Terminal Logger with SOL format
logger = logging.getLogger("SOL.KaggleSyntheticOrchestrator")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import Kaggle API extended wrapper
KAGGLE_AVAILABLE = False
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    logger.warning("Kaggle package not found in synthetic data orchestrator.")

def retry_kaggle_call(func, *args, max_retries=3, initial_delay=2, backoff_factor=2, **kwargs):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error("Kaggle API call %s failed after %d attempts: %s", func.__name__, max_retries, e)
                raise e
            logger.warning(
                "Kaggle API call %s failed (attempt %d/%d). Retrying in %ds... Error: %s",
                func.__name__, attempt, max_retries, delay, e
            )
            time.sleep(delay)
            delay *= backoff_factor

class KaggleSyntheticOrchestrator:
    """
    تدير دورة حياة تشغيل مهمة توليد البيانات الاصطناعية على سحابة كاجل:
    تتحقق من الحساب، ترفع الملف الأصلي وإعدادات التوليد، تبدأ الـ Kernel، وتراقب التقدم.
    """
    def __init__(self, username: str = None, api_token: str = None):
        self.username = username or os.environ.get("KAGGLE_USERNAME", "").strip()
        self.api_token = api_token or os.environ.get("KAGGLE_API_TOKEN", "").strip() or os.environ.get("KAGGLE_KEY", "").strip()
        self.api = None
        self.is_running = False
        
        if KAGGLE_AVAILABLE and self.username and self.api_token:
            try:
                self.api = self._authenticate()
            except Exception as e:
                logger.error("Kaggle Auth Initialization failed in synthetic orchestrator: %s", e)

    def _authenticate(self) -> "KaggleApi":
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

        if orig_user is not None:
            os.environ["KAGGLE_USERNAME"] = orig_user
        if orig_key is not None:
            os.environ["KAGGLE_KEY"] = orig_key

        try:
            config = getattr(api, "config", None) or getattr(api, "configuration", None)
            if not config and hasattr(api, "api_client") and api.api_client:
                config = getattr(api.api_client, "configuration", None)
            if config:
                config.connection_timeout = 60
                config.read_timeout = 60
        except Exception as config_err:
            logger.warning("Could not set custom Kaggle API config timeout: %s", config_err)

        return api

    def run_remote_generate(
        self,
        task_id: str,
        dataset_id: str,
        num_rows: int,
        model_type: str,
        epochs: int = 30,
        null_pct: float = 0.0,
        outlier_pct: float = 0.0
    ):
        from backend.store import _store_tasks, _store_parquet_path, _store_filename
        from backend.utils.bundler import create_studio_bundle
        
        if not self.api:
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = "Kaggle API credentials are not configured or failed auth. Please verify in Settings."
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
                
            safe_username = self.username.lower() if self.username else ""
            clean_uuid = "".join(c for c in task_id if c.isalnum()).lower()
            dataset_slug = f"solsynthin{clean_uuid}"
            kernel_slug = f"solsynthker{clean_uuid}"
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
            
            # Write configuration options
            config_payload = {
                "dataset_id": dataset_id,
                "filename": _store_filename.get(dataset_id, "dataset.csv"),
                "num_rows": num_rows,
                "model_type": model_type,
                "epochs": epochs,
                "null_pct": null_pct,
                "outlier_pct": outlier_pct,
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
                    status_str = str(status_data).lower()
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
            root_kernel_path = Path(__file__).resolve().parent.parent.parent.parent / "kaggle_synthetic_kernel.py"
            with open(root_kernel_path, "r", encoding="utf-8") as f:
                kernel_code = f.read()
                
            # Replace placeholder variables
            kernel_code = kernel_code.replace('__KAGGLE_USERNAME_PLACEHOLDER__', self.username)
            kernel_code = kernel_code.replace('__KAGGLE_KEY_PLACEHOLDER__', self.api_token)
            kernel_code = kernel_code.replace('__GROQ_API_KEY_PLACEHOLDER__', os.environ.get("GROQ_API_KEY", ""))
            kernel_code = kernel_code.replace('__DATASET_REF_PLACEHOLDER__', full_dataset_ref)
            
            with open(temp_kernel_dir / "kaggle_synthetic_kernel.py", "w", encoding="utf-8") as f:
                f.write(kernel_code)
            
            # Kernel metadata file for Kaggle (mounts the dataset)
            kernel_meta = {
                "id": full_kernel_ref,
                "title": kernel_slug,
                "code_file": "kaggle_synthetic_kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,  # Enable GPU for CTGAN/TVAE training!
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
            
            kernel_url = f"https://www.kaggle.com/code/{safe_username}/{kernel_slug}"
            _store_tasks[task_id]["kernel_url"] = kernel_url
            
            _store_tasks[task_id]["progress"] = 60
            
            # Run asynchronous polling thread
            threading.Thread(
                target=self._poll_kernel_status,
                args=(task_id, dataset_id, full_kernel_ref, full_dataset_ref),
                daemon=True
            ).start()
            
        except Exception as e:
            self.is_running = False
            logger.error("Failed to trigger remote synthetic generation on Kaggle: %s", e)
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = f"Submission error: {str(e)}"
            
        finally:
            if temp_dataset_dir and temp_dataset_dir.exists():
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)
            if temp_kernel_dir and temp_kernel_dir.exists():
                shutil.rmtree(temp_kernel_dir, ignore_errors=True)

    def _poll_kernel_status(
        self,
        task_id: str,
        dataset_id: str,
        kernel_ref: str,
        dataset_ref: str
    ):
        from backend.store import _store_tasks
        
        try:
            max_poll_seconds = 7200  # 2 hours limit
            poll_interval = 5
            elapsed = 0
            
            logger.info("Remote monitoring thread active for synthetic kernel '%s'...", kernel_ref)
            while elapsed < max_poll_seconds:
                status = self._get_kernel_status(kernel_ref)
                
                if status == "complete":
                    self._download_and_inject(task_id, dataset_id, kernel_ref, dataset_ref)
                    return
                elif status == "error":
                    _store_tasks[task_id]["status"] = "failed"
                    _store_tasks[task_id]["error"] = "Kaggle execution failed on remote server."
                    self._safe_cleanup_remote(dataset_ref)
                    return
                    
                progress_offset = min(60 + int((elapsed / 600) * 33), 93)  # Slow progress up to 93%
                if task_id in _store_tasks:
                    _store_tasks[task_id]["progress"] = progress_offset
                    
                time.sleep(poll_interval)
                elapsed += poll_interval
                
            _store_tasks[task_id]["status"] = "failed"
            _store_tasks[task_id]["error"] = "Kaggle kernel run timed out."
            self._safe_cleanup_remote(dataset_ref)
        finally:
            self.is_running = False

    def _get_kernel_status(self, kernel_ref: str) -> str:
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
            logger.error("Error querying remote status for synthetic kernel '%s': %s", kernel_ref, e)
            return "error"

    def _download_and_inject(
        self,
        task_id: str,
        dataset_id: str,
        kernel_ref: str,
        dataset_ref: str
    ):
        import pandas as pd
        from backend.store import (
            _store, _store_parquet_path, _store_tasks, _store_filename
        )
        
        logger.info("Remote job successful. Downloading outputs from Kaggle...")
        
        temp_out_dir = Path("temp_snapshots") / f"out_synthetic_{task_id}"
        temp_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            retry_kaggle_call(
                self.api.kernels_output,
                kernel_ref,
                path=str(temp_out_dir),
                force=True
            )
            
            synthetic_parquet_path = temp_out_dir / "synthetic.parquet"
            report_json_path = temp_out_dir / "report.json"
            privacy_json_path = temp_out_dir / "privacy_report.json"
            data_dict_md_path = temp_out_dir / "data_dict.md"
            
            if not synthetic_parquet_path.exists():
                log_content = ""
                try:
                    log_files = list(temp_out_dir.glob("*.log"))
                    if log_files:
                        with open(log_files[0], "r", encoding="utf-8") as lf:
                            log_content = lf.read()
                except Exception as log_err:
                    log_content = f"Could not read log file: {log_err}"
                
                if log_content:
                    raise FileNotFoundError(f"Outputs folder is missing 'synthetic.parquet'. Remote Kernel Log:\n{log_content}")
                else:
                    raise FileNotFoundError("Outputs folder is missing 'synthetic.parquet'. No remote logs found.")
                
            orig_parquet_path = _store_parquet_path.get(dataset_id)
            if not orig_parquet_path:
                orig_parquet_path = str(Path("temp_snapshots") / f"{dataset_id}.parquet")
                
            # Path to save synthetic data locally
            synthetic_id = f"synthetic_{dataset_id}_{int(time.time())}"
            final_synthetic_path = orig_parquet_path.replace(".parquet", f"_synthetic_{task_id}.parquet")
            shutil.move(str(synthetic_parquet_path), final_synthetic_path)
            
            # Read df
            synthetic_df = pd.read_parquet(final_synthetic_path)
            
            # Save to global store
            _store[synthetic_id] = synthetic_df if len(synthetic_df) <= 50000 else synthetic_df.head(500)
            _store_parquet_path[synthetic_id] = final_synthetic_path
            
            orig_filename = _store_filename.get(dataset_id, "dataset.csv")
            base_filename = orig_filename.rsplit('.', 1)[0]
            _store_filename[synthetic_id] = f"synthetic_{base_filename}.csv"
            
            # Load fidelity report
            if report_json_path.exists():
                with open(report_json_path, "r", encoding="utf-8") as f:
                    fidelity_report = json.load(f)
            else:
                fidelity_report = []
                
            # Load privacy report
            if privacy_json_path.exists():
                with open(privacy_json_path, "r", encoding="utf-8") as f:
                    privacy_report = json.load(f)
            else:
                privacy_report = {"privacy_score": 0.0, "risk_level": "Unknown", "rows": []}
                
            # Load data dictionary
            if data_dict_md_path.exists():
                with open(data_dict_md_path, "r", encoding="utf-8") as f:
                    data_dict = f.read()
            else:
                data_dict = "# Data Dictionary not generated."
                
            # Preview records (up to 100 rows)
            preview_df = synthetic_df.head(100).replace({np.nan: None})
            preview_records = json.loads(preview_df.to_json(orient="records"))
            
            # Finalize status mapping
            _store_tasks[task_id]["progress"] = 100
            _store_tasks[task_id]["status"] = "completed"
            _store_tasks[task_id]["result"] = {
                "dataset_id": dataset_id,
                "synthetic_dataset_id": synthetic_id,
                "synthetic_filename": _store_filename[synthetic_id],
                "stats": {
                    "total_rows": len(synthetic_df),
                    "total_cols": len(synthetic_df.columns),
                    "fidelity_avg": sum(r["fidelity_score"] for r in fidelity_report) / len(fidelity_report) if fidelity_report else 0
                },
                "fidelity_report": fidelity_report,
                "privacy_report": privacy_report,
                "data_dict": data_dict,
                "preview_records": preview_records
            }
            logger.info("Successfully loaded and injected Kaggle synthetic outputs for task '%s'.", task_id)
            
        except Exception as e:
            err_msg = str(e)
            if "sdv" in err_msg or "الإنترنت" in err_msg or "Internet" in err_msg:
                logger.error("⚠️ WARNING: Kaggle execution failed because sdv package is missing. Internet access might be disabled on Kaggle Cloud.")
                user_friendly_err = (
                    "تنبيه: فشل التشغيل بسبب تعذر تثبيت مكتبة sdv على كاجل (عدم وجود اتصال بالإنترنت). "
                    "يرجى تفعيل خيار الإنترنت (Internet) في إعدادات حساب كاجل الخاص بك "
                    "(عبر التحقق من رقم الهاتف وتفعيل خيار 'Internet' في لوحة الإعدادات الجانبية للنوت بوك) "
                    "أو استخدام النموذج الأساسي Basic (Statistical) للتوليد بدون إنترنت."
                )
                _store_tasks[task_id]["status"] = "failed"
                _store_tasks[task_id]["error"] = f"{user_friendly_err}\n\n[تفاصيل الخطأ]:\n{err_msg}"
            else:
                logger.error("Error unpacking downloaded synthetic artifacts: %s", e)
                _store_tasks[task_id]["status"] = "failed"
                _store_tasks[task_id]["error"] = f"Extraction failure: {err_msg}"
            
        finally:
            shutil.rmtree(temp_out_dir, ignore_errors=True)
            self._safe_cleanup_remote(dataset_ref)

    def _safe_cleanup_remote(self, dataset_ref: str):
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
            logger.info("[SUCCESS] Kaggle temp synthetic dataset deleted.")
        except Exception as e:
            logger.warning("Could not execute remote cleanup on synthetic dataset '%s': %s", dataset_ref, e)
