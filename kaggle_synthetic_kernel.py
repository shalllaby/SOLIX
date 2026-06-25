#!/usr/bin/env python3
"""
kaggle_synthetic_kernel.py - Remote Execution Kernel for SOL Platform Synthetic Data Studio.

This script runs on Kaggle's infrastructure. It:
1. Waits for the mounted dataset to propagate.
2. Locates raw.parquet and config.json configurations.
3. Dynamically installs 'sdv' if not present in the environment.
4. Modifies sys.path to permit imports of packaged code from studio_core folder.
5. Executes profiling and the selected synthetic generation engine (Basic, GaussianCopula, TVAE, CTGAN, or Benchmark).
6. Applies noise and outlier injections.
7. Computes fidelity score comparison, privacy DCR score, and data dictionary.
8. Saves all resulting output files to /kaggle/working/.
"""

# Credentials and dataset reference placeholders to be dynamically injected by orchestrator
KAGGLE_USERNAME = "__KAGGLE_USERNAME_PLACEHOLDER__"
KAGGLE_KEY = "__KAGGLE_KEY_PLACEHOLDER__"
GROQ_API_KEY = "__GROQ_API_KEY_PLACEHOLDER__"
DATASET_REF = "__DATASET_REF_PLACEHOLDER__"

import os
import sys
import subprocess

# Configure environment variables
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY
if GROQ_API_KEY and not GROQ_API_KEY.startswith("__"):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

import json
import time
import traceback
import gc

def print_log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp}] [INFO] [SOL.KaggleSyntheticKernel] {msg}")
    sys.stdout.flush()

# Check and install sdv
def has_internet():
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("pypi.org", 80))
        return True
    except Exception:
        return False

try:
    print_log("Checking sdv installation...")
    import sdv
    print_log(f"sdv version {sdv.__version__} is already installed.")
except ImportError:
    if not has_internet():
        print_log("[WARNING] No internet access detected on Kaggle Cloud. Skipping pip install sdv. Please enable 'Internet' in the Kaggle kernel settings panel if deep learning models (CTGAN, TVAE) are required.")
    else:
        print_log("sdv is not installed. Auto-installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sdv", "--retries", "1", "--timeout", "10"])
            import sdv
            print_log(f"[SUCCESS] sdv version {sdv.__version__} installed successfully.")
        except Exception as e:
            print_log(f"[ERROR] Failed to install sdv: {e}")
            # We don't exit here, in case only "basic" method is used which doesn't need sdv.

def wait_for_dataset():
    print_log("Waiting for mounted dataset (raw.parquet) under /kaggle/input/...")
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'raw.parquet' in files:
            print_log(f"Dataset found instantly at: {root}")
            return root
            
    max_retries = 60
    for attempt in range(max_retries):
        print_log(f"Mount not ready (attempt {attempt+1}/{max_retries}). Waiting 2s...")
        time.sleep(2)
        
        found_path = None
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'raw.parquet' in files:
                found_path = root
                break
        
        if found_path:
            print_log(f"Dataset found at: {found_path}")
            return found_path
        
    raise FileNotFoundError("[ERROR] Mount timeout. 'raw.parquet' never appeared in /kaggle/input.")

def main():
    print_log("=== REMOTE KAGGLE SYNTHETIC GENERATION KERNEL STARTING ===")
    
    try:
        # 1. Wait for mounted files
        dataset_dir = wait_for_dataset()
        
        # 2. Setup Paths
        source_code_dir = os.path.join(dataset_dir, 'studio_core')
        raw_parquet_path = os.path.join(dataset_dir, 'raw.parquet')
        config_json_path = os.path.join(dataset_dir, 'config.json')
        
        output_dir = "/kaggle/working"
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. Configure import paths
        if os.path.exists(source_code_dir):
            sys.path.insert(0, source_code_dir)
            sys.path.insert(0, os.path.join(source_code_dir, "backend/tools"))
            print_log(f"[SUCCESS] Appended paths from: '{source_code_dir}' to sys.path.")
        else:
            print_log(f"[WARNING] 'studio_core' directory not found in {dataset_dir}. Imports may fail.")
            
        # 4. Load Data & Config
        if not os.path.exists(raw_parquet_path):
            print_log(f"[ERROR] Raw dataset '{raw_parquet_path}' does not exist.")
            sys.exit(1)
            
        if not os.path.exists(config_json_path):
            print_log(f"[ERROR] Configuration file '{config_json_path}' does not exist.")
            sys.exit(1)
            
        print_log("Loading raw dataset and strategy configurations...")
        import pandas as pd
        raw_df = pd.read_parquet(raw_parquet_path)
        
        with open(config_json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        print_log(f"[SUCCESS] Loaded raw dataset: Shape = {raw_df.shape}")
        
        # Inject Groq API Key if provided in config
        groq_api_key = config.get("groq_api_key")
        if groq_api_key:
            print_log("Injecting GROQ_API_KEY into remote runtime environment.")
            os.environ["GROQ_API_KEY"] = groq_api_key
            
        # 5. Import Synthetic Data Engine
        print_log("Importing synthetic engine functions...")
        from synthetic_data.engine import (
            profile_dataframe,
            generate_synthetic,
            generate_synthetic_fast,
            generate_synthetic_tvae,
            generate_synthetic_ai,
            inject_noise,
            generate_report,
            generate_privacy_report,
            generate_data_dictionary
        )
        
        model_type = config.get("model_type", "basic")
        num_rows = int(config.get("num_rows", 100))
        epochs = int(config.get("epochs", 30))
        null_pct = float(config.get("null_pct", 0.0))
        outlier_pct = float(config.get("outlier_pct", 0.0))
        
        # 6. Profile Dataframe
        print_log("Analyzing dataset schema and profiling data statistics...")
        profile = profile_dataframe(raw_df)
        print_log("[SUCCESS] Profile constructed successfully.")
        
        # 7. Generate Synthetic Dataset
        print_log(f"Starting generation using model: {model_type} (Target rows: {num_rows})...")
        
        def progress_cb(pct, msg):
            print_log(f"[Progress {int(pct*100)}%] {msg}")
            
        if model_type == "basic":
            synthetic_df = generate_synthetic(raw_df, profile, num_rows)
        elif model_type == "gaussian_copula":
            synthetic_df = generate_synthetic_fast(raw_df, profile, num_rows, progress_callback=progress_cb)
        elif model_type == "tvae":
            synthetic_df = generate_synthetic_tvae(raw_df, profile, num_rows, epochs=epochs, progress_callback=progress_cb)
        elif model_type == "ctgan":
            synthetic_df = generate_synthetic_ai(raw_df, profile, num_rows, epochs=epochs, progress_callback=progress_cb)
        elif model_type == "benchmark":
            print_log("Auto-Benchmarking multiple models...")
            df_basic = generate_synthetic(raw_df, profile, num_rows)
            score_basic = generate_report(raw_df, df_basic, profile)["fidelity_score"].mean()
            print_log(f"Basic model Fidelity Score: {score_basic:.2f}%")
            
            try:
                df_copula = generate_synthetic_fast(raw_df, profile, num_rows)
                score_copula = generate_report(raw_df, df_copula, profile)["fidelity_score"].mean()
                print_log(f"Gaussian Copula Fidelity Score: {score_copula:.2f}%")
            except Exception as e:
                print_log(f"Gaussian Copula failed: {e}")
                df_copula = None
                score_copula = -1
                
            if score_copula > score_basic and df_copula is not None:
                synthetic_df = df_copula
                print_log("Selected model: Gaussian Copula")
            else:
                synthetic_df = df_basic
                print_log("Selected model: Basic Statistical Model")
        else:
            raise ValueError(f"Invalid model type: {model_type}")
            
        print_log(f"[SUCCESS] Generation complete. Generated shape: {synthetic_df.shape}")
        
        # 8. Inject Noise
        if null_pct > 0 or outlier_pct > 0:
            print_log(f"Injecting noise: Nulls = {null_pct}%, Outliers = {outlier_pct}%...")
            synthetic_df = inject_noise(synthetic_df, profile, null_pct, outlier_pct)
            print_log("[SUCCESS] Noise injection completed.")
            
        # 9. Compute Reports
        print_log("Generating fidelity report...")
        report_df = generate_report(raw_df, synthetic_df, profile)
        fidelity_report = report_df.to_dict(orient="records")
        print_log(f"Average Fidelity Score: {report_df['fidelity_score'].mean():.2f}%")
        
        print_log("Generating privacy report (DCR metric)...")
        privacy_rows, privacy_metrics = generate_privacy_report(raw_df, synthetic_df, profile)
        print_log(f"Privacy Score: {privacy_metrics['privacy_score']}% (Risk Level: {privacy_metrics['risk_level']})")
        
        print_log("Generating data dictionary...")
        data_dict = generate_data_dictionary(profile, synthetic_df)
        
        # 10. Save Output Artifacts
        synthetic_parquet_out = os.path.join(output_dir, "synthetic.parquet")
        report_json_out = os.path.join(output_dir, "report.json")
        privacy_json_out = os.path.join(output_dir, "privacy_report.json")
        data_dict_md_out = os.path.join(output_dir, "data_dict.md")
        
        print_log(f"Saving synthetic dataset to: '{synthetic_parquet_out}'...")
        synthetic_df.to_parquet(synthetic_parquet_out, index=False)
        
        print_log(f"Saving report details to: '{report_json_out}'...")
        with open(report_json_out, "w", encoding="utf-8") as f:
            json.dump(fidelity_report, f, indent=4, ensure_ascii=False)
            
        print_log(f"Saving privacy details to: '{privacy_json_out}'...")
        with open(privacy_json_out, "w", encoding="utf-8") as f:
            json.dump(privacy_metrics, f, indent=4, ensure_ascii=False)
            
        print_log(f"Saving data dictionary to: '{data_dict_md_out}'...")
        with open(data_dict_md_out, "w", encoding="utf-8") as f:
            f.write(data_dict)
            
        print_log("[SUCCESS] All job outputs written to working directory.")
        
    except Exception as e:
        print_log(f"[ERROR] Execution failed: {e}")
        print_log(traceback.format_exc())
        sys.exit(1)
        
    gc.collect()
    print_log("=== REMOTE KAGGLE SYNTHETIC GENERATION KERNEL FINISHED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
