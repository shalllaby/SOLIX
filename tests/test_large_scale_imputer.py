import sys
import os
import numpy as np
import pandas as pd
import time
import psutil

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_imputer import AIImputer
from core.cleaner import SmartDataCleaner

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def run_large_scale_test():
    print("=== STARTING LARGE-SCALE IMPUTER TEST ===")
    
    # 1. Generate a mock dataset of 300,000 rows to trigger:
    # - Representative downsampling (limit is 50,000)
    # - Chunked predictions (limit is 250,000)
    n_rows = 300000
    print(f"Generating mock dataset with {n_rows} rows...")
    
    np.random.seed(42)
    x1 = np.random.uniform(10, 100, n_rows)
    # Highly correlated feature
    x2 = x1 * 2.5 + np.random.normal(0, 5, n_rows)
    
    # Categorical feature
    categories = ['Red', 'Green', 'Blue', 'Yellow']
    cat = np.random.choice(categories, n_rows)
    
    # Numeric column with missing values
    y_numeric = x1 * 1.5 + np.random.normal(0, 2, n_rows)
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'category': cat,
        'target_num': y_numeric
    })
    
    # Force duplicate index values to test index safety
    df.index = [i % 100 for i in range(n_rows)]
    
    # Inject 40% missing values in target_num (120,000 missing, 180,000 training rows)
    missing_mask = np.random.rand(n_rows) < 0.4
    df.loc[missing_mask, 'target_num'] = np.nan
    
    initial_nulls = df['target_num'].isna().sum()
    print(f"Initial missing values in target_num: {initial_nulls} ({initial_nulls/n_rows:.1%})")
    
    mem_before = get_memory_usage()
    print(f"Memory usage before initialization: {mem_before:.2f} MB")
    
    t_start = time.time()
    
    # Initialize AIImputer (should not copy df)
    imputer = AIImputer(df)
    
    mem_after_init = get_memory_usage()
    print(f"Memory usage after imputer init: {mem_after_init:.2f} MB")
    # Verify no massive memory copy happened at initialization
    assert (mem_after_init - mem_before) < 50.0, "Warning: Memory spike detected on imputer init!"
    
    # Run imputation
    print("Running predict_missing_values (should downsample training and run in chunks)...")
    df, imp_status = imputer.predict_missing_values(target_col='target_num', strategy={})
    
    t_end = time.time()
    mem_after_impute = get_memory_usage()
    
    print(f"Imputation completed in {t_end - t_start:.2f} seconds.")
    print(f"Memory usage after imputation: {mem_after_impute:.2f} MB")
    
    final_nulls = df['target_num'].isna().sum()
    print(f"Final missing values in target_num: {final_nulls}")
    
    assert final_nulls == 0, "Error: Some missing values were not imputed!"
    print("✅ Successfully imputed all missing values without OOM!")
    
    # Test entire SmartDataCleaner wrapper with the same dataset
    print("\nTesting SmartDataCleaner pipeline integration...")
    df_cleaner = df.copy()
    df_cleaner.loc[np.random.rand(n_rows) < 0.1, 'target_num'] = np.nan
    
    cleaner = SmartDataCleaner(df_cleaner)
    strategy = {
        "cleaning_strategy": {
            "target_num": "smart_impute"
        }
    }
    
    cleaned_df, report = cleaner.execute_strategy(strategy)
    print(f"Cleaner actions: {report['actions']}")
    assert cleaned_df['target_num'].isna().sum() == 0, "Error: SmartDataCleaner failed to impute all values!"
    print("✅ SmartDataCleaner integration test passed successfully!")

if __name__ == "__main__":
    run_large_scale_test()
