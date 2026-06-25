"""
tests/test_predictive_imputation.py
===================================
Unit test for testing the advanced predictive imputation helper on mixed numeric
and categorical data.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.copilot.imputer import advanced_impute

def test_predictive_imputation():
    # 1. Create a synthetic dataset with numeric correlation and categorical groupings
    np.random.seed(42)
    n_samples = 150
    
    # x1 and x2 are highly correlated
    x1 = np.random.uniform(10, 100, n_samples)
    x2 = x1 * 2.5 + np.random.normal(0, 10, n_samples)
    
    # Categorical feature
    categories = ['Low', 'Medium', 'High']
    cat = np.random.choice(categories, n_samples)
    
    # Integer metric
    age = np.random.randint(18, 80, n_samples)
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'category': cat,
        'age': age
    })
    
    # Cast category column to object
    df['category'] = df['category'].astype('object')
    
    # 2. Inject artificial nulls (NaN)
    df.loc[10:30, 'x1'] = np.nan
    df.loc[25:45, 'x2'] = np.nan
    df.loc[40:60, 'category'] = np.nan
    df.loc[55:75, 'age'] = np.nan
    
    assert df.isnull().any().any(), "Dataframe must contain missing values before test"
    
    # 3. Test AUTO mode
    df_auto = advanced_impute(df, method="auto")
    assert not df_auto.isnull().any().any(), "AUTO mode failed: null values remaining"
    assert df_auto['category'].dtype == 'object', "AUTO mode failed: category column type changed"
    assert pd.api.types.is_integer_dtype(df_auto['age'].dtype), "AUTO mode failed: age column was not restored to integer type"
    
    # Verify categories are valid original categories
    unique_cats = df_auto['category'].dropna().unique()
    for u in unique_cats:
        assert u in categories, f"Invalid category decoded: {u}"

    # 4. Test KNN mode explicitly
    df_knn = advanced_impute(df, method="knn")
    assert not df_knn.isnull().any().any(), "KNN mode failed: null values remaining"

    # 5. Test ITERATIVE (MICE) mode explicitly
    df_mice = advanced_impute(df, method="iterative")
    assert not df_mice.isnull().any().any(), "Iterative (MICE) mode failed: null values remaining"

    # 6. Test SIMPLE mode fallback
    df_simple = advanced_impute(df, method="simple")
    assert not df_simple.isnull().any().any(), "Simple mode failed: null values remaining"

    print("✅ All predictive imputation tests completed successfully!")

if __name__ == "__main__":
    test_predictive_imputation()
