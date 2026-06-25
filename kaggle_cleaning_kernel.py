#!/usr/bin/env python3
"""
kaggle_cleaning_kernel.py - Remote Execution Kernel for Al-Dalil Data Cleaning Studio.

This script runs on Kaggle's infrastructure. It:
1. Waits for the mounted dataset to propagate.
2. Locates raw.parquet and config.json configurations.
3. Modifies sys.path to permit imports of packaged code from the auto-unzipped studio_core folder.
4. Executes cleaning strategy using SmartDataCleaner.
5. Computes quality score updates, cell deltas, and reasoning via AuditReportBuilder.
6. Builds interactive distribution/comparison visual payloads via VizEngine.
7. Saves all resulting output files to /kaggle/working/.
"""

# Credentials and dataset reference placeholders to be dynamically injected by orchestrator
KAGGLE_USERNAME = "__KAGGLE_USERNAME_PLACEHOLDER__"
KAGGLE_KEY = "__KAGGLE_KEY_PLACEHOLDER__"
GROQ_API_KEY = "__GROQ_API_KEY_PLACEHOLDER__"
DATASET_REF = "__DATASET_REF_PLACEHOLDER__"

import os
import sys

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
    print(f"[{timestamp}] [INFO] [SOL.KaggleKernel] {msg}")
    sys.stdout.flush()

def wait_for_dataset():
    print_log("Waiting for mounted dataset (raw.parquet) under /kaggle/input/...")
    # Instant initial check before starting any retry sleep loop
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
    print_log("=== REMOTE KAGGLE CLEANING KERNEL STARTING (AUTO-UNZIP MODE) ===")
    
    try:
        # 1. Wait for DFS Propagation
        dataset_dir = wait_for_dataset()
        
        # 2. Setup Paths (Kaggle already unzipped studio_core.zip into 'studio_core' folder)
        source_code_dir = os.path.join(dataset_dir, 'studio_core')
        raw_parquet_path = os.path.join(dataset_dir, 'raw.parquet')
        config_json_path = os.path.join(dataset_dir, 'config.json')
        
        # Ensure working directory exists for outputs
        output_dir = "/kaggle/working"
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. Configure import paths
        if os.path.exists(source_code_dir):
            sys.path.insert(0, source_code_dir)
            sys.path.insert(0, os.path.join(source_code_dir, "backend/tools"))
            print_log(f"[SUCCESS] Appended paths from: '{source_code_dir}' to sys.path.")
        else:
            print_log(f"[WARNING] 'studio_core' directory not found in {dataset_dir}. Imports may fail.")
            
        # 4. Load Data & Config Snapshots
        if not os.path.exists(raw_parquet_path):
            print_log(f"[ERROR] Raw dataset '{raw_parquet_path}' does not exist.")
            sys.exit(1)
            
        if not os.path.exists(config_json_path):
            print_log(f"[ERROR] Strategy config '{config_json_path}' does not exist.")
            sys.exit(1)
            
        print_log("Loading raw dataset and strategy configurations...")
        import pandas as pd
        raw_df = pd.read_parquet(raw_parquet_path)
        
        with open(config_json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        print_log(f"[SUCCESS] Loaded raw dataset: Shape = {raw_df.shape}")
        
        # 5. Execute Data Cleaning
        print_log("Importing SmartDataCleaner engine...")
        from core.cleaner import SmartDataCleaner
        
        # Inject Groq API Key if provided in config
        groq_api_key = config.get("groq_api_key")
        if groq_api_key:
            print_log("Injecting GROQ_API_KEY into remote runtime environment.")
            os.environ["GROQ_API_KEY"] = groq_api_key
            
        strategy_json = config["strategy_json"]
        policy_config = config.get("policy_config")
        
        print_log("Starting data cleaning execution...")
        cleaner = SmartDataCleaner(raw_df, policy_config=policy_config)
        
        # Cleaner execute returns (cleaned_df, report)
        result = cleaner.execute_strategy(strategy_json)
        cleaned_df, report = result if isinstance(result, tuple) else (result, {"actions": []})
        
        print_log(f"[SUCCESS] Data cleaning completed: Cleaned Shape = {cleaned_df.shape}")
        
        # 6. Build Audit Log
        try:
            print_log("Importing AuditReportBuilder...")
            from audit.engine import AuditReportBuilder
            
            strategy_level = config["strategy_level"]
            filename = config.get("filename", "dataset.csv")
            user_goal = config.get("user_goal", "")
            dataset_id = config["dataset_id"]
            
            print_log("Building structured audit report and quality metrics...")
            audit_builder = AuditReportBuilder(
                raw_df=raw_df,
                cleaned_df=cleaned_df,
                cleaner_report=report,
                strategy_used=strategy_level,
                filename=filename,
                user_goal=user_goal,
                dataset_id=dataset_id,
                strategy_json=strategy_json,
            )
            audit_log = audit_builder.build()
            print_log("[SUCCESS] Audit report built successfully.")
        except Exception as e:
            print_log(f"[ERROR] Audit report generation failed: {e}")
            print_log(traceback.format_exc())
            audit_log = {
                "error": f"Failed to generate audit: {str(e)}",
                "dataset_id": config.get("dataset_id"),
                "filename": config.get("filename", "dataset.csv")
            }
            
        # 7. Generate Visualizations (VizEngine)
        try:
            print_log("Importing VizEngine visualization processor...")
            from viz_engine.engine import VizEngine
            
            print_log("Generating distribution and comparative charts (Plotly figures)...")
            # Sample to 10k rows for visualization performance
            viz_cmp = VizEngine(raw_df=raw_df.head(10000), cleaned_df=cleaned_df.head(10000))
            viz_payload = viz_cmp.comparison()
            print_log("[SUCCESS] Visualization figures generated successfully.")
        except Exception as e:
            print_log(f"[ERROR] Visualization generation failed: {e}")
            print_log(traceback.format_exc())
            viz_payload = {"error": f"Failed to generate visuals: {str(e)}", "mode": "comparison"}
            
        # 8. Serialize Output Artifacts
        cleaned_parquet_out = os.path.join(output_dir, "cleaned.parquet")
        audit_json_out = os.path.join(output_dir, "audit.json")
        viz_json_out = os.path.join(output_dir, "viz_payload.json")
        
        print_log(f"Saving cleaned dataset to: '{cleaned_parquet_out}'...")
        cleaned_df.to_parquet(cleaned_parquet_out, index=False)
        
        print_log(f"Saving audit log to: '{audit_json_out}'...")
        with open(audit_json_out, "w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=4, ensure_ascii=False)
            
        print_log(f"Saving visualization payloads to: '{viz_json_out}'...")
        with open(viz_json_out, "w", encoding="utf-8") as f:
            json.dump(viz_payload, f, indent=4)
            
        print_log("[SUCCESS] All job outputs written to working directory.")
        
    except Exception as e:
        print_log(f"[ERROR] Execution failed: {e}")
        print_log(traceback.format_exc())
        sys.exit(1)
        
    # Trigger garbage collection
    gc.collect()
    print_log("=== REMOTE KAGGLE CLEANING KERNEL FINISHED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
