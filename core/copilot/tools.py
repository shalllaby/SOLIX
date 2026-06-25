import pandas as pd
from core.cleaner import SmartDataCleaner

def clean_column_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using Z-score and automatically 
    impute the missing values using the memory-safe AIImputer.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    cleaner = SmartDataCleaner(df)
    strategy = {"cleaning_strategy": {col: "remove_outliers"}}
    df_clean, report = cleaner.execute_strategy(strategy)
    if not hasattr(df_clean, "attrs"):
        df_clean.attrs = {}
    existing_logs = df_clean.attrs.get("audit_log", [])
    if not isinstance(existing_logs, list):
        existing_logs = []
    for act in report.get("actions", []):
        existing_logs.append({
            "column": col,
            "issue": "Outliers detected",
            "resolution": act
        })
    df_clean.attrs["audit_log"] = existing_logs
    return df_clean

def smart_impute_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Impute missing values in a column using the memory-safe AIImputer 
    (RandomForest + 1D index slicing + representative downsampling + concurrent chunking).
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    cleaner = SmartDataCleaner(df)
    strategy = {"cleaning_strategy": {col: "smart_impute"}}
    df_clean, report = cleaner.execute_strategy(strategy)
    if not hasattr(df_clean, "attrs"):
        df_clean.attrs = {}
    existing_logs = df_clean.attrs.get("audit_log", [])
    if not isinstance(existing_logs, list):
        existing_logs = []
    for act in report.get("actions", []):
        existing_logs.append({
            "column": col,
            "issue": "Missing values",
            "resolution": act
        })
    df_clean.attrs["audit_log"] = existing_logs
    return df_clean

def standardize_column_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Standardize datetime formats in a column.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    cleaner = SmartDataCleaner(df)
    strategy = {"cleaning_strategy": {col: "standardize_date"}}
    df_clean, report = cleaner.execute_strategy(strategy)
    if not hasattr(df_clean, "attrs"):
        df_clean.attrs = {}
    existing_logs = df_clean.attrs.get("audit_log", [])
    if not isinstance(existing_logs, list):
        existing_logs = []
    for act in report.get("actions", []):
        existing_logs.append({
            "column": col,
            "issue": "Inconsistent date format",
            "resolution": act
        })
    df_clean.attrs["audit_log"] = existing_logs
    return df_clean

def fuzzy_fix_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Fuzzy merge text typos and spelling variations in a text column.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    cleaner = SmartDataCleaner(df)
    strategy = {"cleaning_strategy": {col: "fuzzy_fix"}}
    df_clean, report = cleaner.execute_strategy(strategy)
    if not hasattr(df_clean, "attrs"):
        df_clean.attrs = {}
    existing_logs = df_clean.attrs.get("audit_log", [])
    if not isinstance(existing_logs, list):
        existing_logs = []
    for act in report.get("actions", []):
        existing_logs.append({
            "column": col,
            "issue": "Text variation/typos",
            "resolution": act
        })
    df_clean.attrs["audit_log"] = existing_logs
    return df_clean
