"""
tests/test_tier_cleaners.py
===========================
Unit test to verify fast_clean and deep_clean pipelines.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.copilot.cleaner import fast_clean, deep_clean

def test_fast_and_deep_cleaners():
    # 1. Create a raw messy dataset
    data = {
        'age': [25, 30, 30, np.nan, 40, 200, 25],  # includes missing, duplicate (30), duplicate (25), extreme outlier (200)
        'city': ['Cairo', 'Ciro', 'Ciro', 'Alex', np.nan, 'Alex', 'Cairo'], # spelling typos, missing values, duplicates
        'joined_date': ['2026-01-15', '2026-02-20', '2026-02-20', np.nan, '2026-04-10', '2026-05-01', '2026-01-15'], # dates
        'salary': ['5000', '6000', '6000', '7500', '9000', '12000', '5000'] # string numbers
    }
    df = pd.DataFrame(data)
    
    # 2. Run Fast Clean
    df_fast, fast_audit = fast_clean(df)
    
    # Assertions for fast_clean:
    # - Duplicates are removed (should go from 7 rows to 5 rows)
    assert len(df_fast) == 5, f"Deduplication failed: expected 5 rows, got {len(df_fast)}"
    
    # - salary is coerced to numeric
    assert pd.api.types.is_numeric_dtype(df_fast['salary'].dtype), "Salary coercion failed"
    
    # - Simple imputation is applied (no nulls in age or city)
    assert not df_fast['age'].isnull().any(), "Age simple imputation failed"
    assert not df_fast['city'].isnull().any(), "City simple imputation failed"
    
    # - Datetime columns are auto-parsed and features extracted
    assert 'joined_date_year' in df_fast.columns, "Datetime year extraction failed"
    assert 'joined_date_month' in df_fast.columns, "Datetime month extraction failed"
    assert 'joined_date_is_weekend' in df_fast.columns, "Datetime weekend extraction failed"
    assert df_fast['joined_date_year'].iloc[0] == 2026, "Datetime year value mismatch"

    # 3. Run Deep Clean
    # Re-create a fresh raw dataset to test deep clean independently
    df_raw = pd.DataFrame(data)
    df_deep, deep_audit = deep_clean(df_raw, scaling=False)
    
    # Assertions for deep_clean:
    # - Inherits fast clean deduplication (5 rows remaining)
    assert len(df_deep) == 5
    
    # - Fuzzy matching merges "Ciro" to "Cairo" (since Cairo is dominant and spelling similarity is >85%)
    # Let's verify "Ciro" was renamed to "Cairo"
    assert "Ciro" not in df_deep['city'].values, "Fuzzy text merging failed to merge Ciro"
    assert "Cairo" in df_deep['city'].values, "Fuzzy text merging lost dominant value Cairo"
    
    # - Outlier anomaly detection is applied since we have at least 2 numeric columns (age and salary)
    assert 'is_anomaly' in df_deep.columns, "Outlier flagging failed to add is_anomaly column"
    # The value 200 is an extreme outlier, it should be flagged as an anomaly
    anomaly_rows = df_deep[df_deep['age'] == 200]
    if not anomaly_rows.empty:
        assert anomaly_rows['is_anomaly'].iloc[0] == 1, "Isolation Forest failed to flag extreme outlier"

    print("✅ All fast_clean and deep_clean unit tests completed successfully!")

if __name__ == "__main__":
    test_fast_and_deep_cleaners()
