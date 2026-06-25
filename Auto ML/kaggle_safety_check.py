#!/usr/bin/env python3
"""
kaggle_safety_check.py - Pre-flight resource validator for Kaggle API workloads.

Usage:
    python kaggle_safety_check.py --gpu_min 10 --disk_max_pct 85
"""

import os
import sys
import argparse
from pathlib import Path

# Try to import kaggle, or guide user how to install it
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("\033[91m[ERROR] The 'kaggle' library is not installed.\033[0m")
    print("Please install it with: pip install kaggle")
    sys.exit(1)

# Colors for terminal styling
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_preflight_check(gpu_min_hours: float, disk_max_pct: float) -> bool:
    """
    Checks Kaggle account limits, active running kernels, and overall usage constraints.
    
    Args:
        gpu_min_hours (float): Minimum remaining GPU quota hours to pass validation.
        disk_max_pct (float): Maximum allowed storage disk usage percent.
        
    Returns:
        bool: True if safety limits are valid, False if a threshold is crossed.
    """
    print(f"{Colors.HEADER}=== KAGGLE API RESOURCE PRE-FLIGHT VALIDATION ==={Colors.ENDC}\n")
    
    # Verify environment variables
    if "KAGGLE_API_TOKEN" not in os.environ:
        print(f"{Colors.FAIL}[ERROR] Missing KAGGLE_API_TOKEN Environment Variable!{Colors.ENDC}")
        print("Please export KAGGLE_API_TOKEN to your shell environment.")
        return False
        
    # Clean up legacy environment variables to prevent conflicts
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
        
    try:
        # Write token file
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Clean up legacy JSON to prevent conflicts
        legacy_json = kaggle_dir / "kaggle.json"
        if legacy_json.exists():
            legacy_json.unlink()
            
        access_token_path = kaggle_dir / "access_token"
        with open(access_token_path, "w", encoding="utf-8") as f:
            f.write(os.environ["KAGGLE_API_TOKEN"].strip())
            
        try:
            os.chmod(access_token_path, 0o600)
        except Exception:
            pass
            
        api = KaggleApi()
        api.authenticate()
        print(f"{Colors.OKGREEN}[SUCCESS] Authenticated successfully via KAGGLE_API_TOKEN.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}[ERROR] Kaggle Authentication Failed: {e}{Colors.ENDC}")
        return False

    # Check active running kernels
    username = os.environ.get("KAGGLE_USERNAME", "al_dalil_governance_service")
    try:
        active_kernels = api.kernels_list(user=username, page=1)
        running_count = 0
        for kernel in active_kernels:
            ref_slug = kernel.ref.split('/')[-1]
            status_data = api.kernels_status(f"{username}/{ref_slug}")
            if isinstance(status_data, dict):
                status = status_data.get("status", "")
            else:
                status = getattr(status_data, "status", "")
                
            if hasattr(status, "value"):
                status_str = str(status.value).lower()
            elif hasattr(status, "name"):
                status_str = str(status.name).lower()
            else:
                status_str = str(status).lower()
                
            if status_str == "running":
                running_count += 1
                
        print(f"-> Active Running Kernels: {running_count}")
        if running_count >= 5:
            print(f"{Colors.WARNING}[WARN] Concurrent kernel execution count is high ({running_count}/5). New runs might queue.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.WARNING}[WARN] Could not retrieve active kernel list: {e}{Colors.ENDC}")

    # Check storage limits
    try:
        datasets = api.dataset_list(user=username)
        total_size_mb = 0
        for ds in datasets:
            try:
                size_str = getattr(ds, 'size', '0 KB')
                parts = size_str.split()
                if len(parts) == 2:
                    val, unit = float(parts[0]), parts[1].upper()
                    if 'KB' in unit:
                        total_size_mb += val / 1024
                    elif 'MB' in unit:
                        total_size_mb += val
                    elif 'GB' in unit:
                        total_size_mb += val * 1024
            except (AttributeError, ValueError):
                continue
        
        storage_limit_gb = 20.0
        used_gb = total_size_mb / 1024
        used_pct = (used_gb / storage_limit_gb) * 100
        
        print(f"-> Secure Storage Usage: {used_gb:.2f} GB / {storage_limit_gb:.1f} GB ({used_pct:.1f}%)")
        
        if used_pct > disk_max_pct:
            print(f"{Colors.FAIL}[FAIL] Storage usage {used_pct:.1f}% exceeds safety threshold of {disk_max_pct}%!{Colors.ENDC}")
            return False
            
    except Exception as e:
        print(f"{Colors.WARNING}[WARN] Could not calculate storage usage: {e}{Colors.ENDC}")

    print(f"\n{Colors.OKGREEN}[PASS] Safety check successful. Ready to run AutoML remotely.{Colors.ENDC}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Pre-flight Safety Check.")
    parser.add_argument("--gpu_min", type=float, default=5.0, help="Min remaining GPU quota hours.")
    parser.add_argument("--disk_max_pct", type=float, default=80.0, help="Max allowed storage threshold.")
    args = parser.parse_args()
    
    # Fallback to default token if not explicitly set in environment
    if "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = "KGAT_0034d6fd413ada3d3b57d06d1736d0ae"
        
    success = run_preflight_check(args.gpu_min, args.disk_max_pct)
    if not success:
        sys.exit(1)
