"""
core/copilot/cleaner.py
=======================
Enterprise-grade data cleaning engine for Voice Data Copilot (SOL).
Provides:
  - fast_clean: lightweight deduplication, type coercion, simple imputer, and datetime parsing.
  - deep_clean: inherits fast_clean, runs advanced predictive imputation, flags multivariate outliers via Isolation Forest, and performs fuzzy text spelling consolidation.
"""

import numpy as np
import pandas as pd
import difflib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from core.copilot.imputer import advanced_impute

def _is_time_series(df: pd.DataFrame) -> bool:
    """Detect if the dataframe is a time-series based on index class, columns, and sorted order."""
    # 1. Check if index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    
    # 2. Check for datetime columns
    dt_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    if dt_cols:
        for col in dt_cols:
            col_series = df[col].dropna()
            if not col_series.empty:
                if col_series.is_monotonic_increasing or col_series.is_monotonic_decreasing:
                    return True
                
    # 3. Check for typical column name indicators if monotonically increasing
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['date', 'timestamp', 'time', 'datetime']):
            try:
                # Test parse first 50 values
                sample = df[col].dropna().head(50)
                if not sample.empty:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notnull().mean() > 0.7:
                        if parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing:
                            return True
            except Exception:
                pass
    return False

def _is_id_or_key_column(col_name: str, series: pd.Series) -> bool:
    """Identify if a column represents unique IDs, serial codes, or alphanumeric keys."""
    col_lower = col_name.lower()
    # 1. Name heuristics
    if any(x in col_lower for x in ["id", "key", "code", "num", "serial", "ref", "hash", "pk", "fk"]):
        return True
    # 2. Value pattern heuristics
    non_nulls = series.dropna().astype(str)
    if non_nulls.empty:
        return False
    # If values are unique/high-cardinality and look alphanumeric
    is_alphanumeric = non_nulls.str.match(r'^[A-Za-z0-9\-_]+$').all()
    has_letters = non_nulls.str.contains(r'[A-Za-z]').any()
    has_digits = non_nulls.str.contains(r'\d').any()
    
    if is_alphanumeric and has_letters and has_digits:
        # Check cardinality: if cardinality is high (e.g. > 70% unique values)
        cardinality = len(non_nulls.unique()) / len(non_nulls)
        if cardinality > 0.7:
            return True
    return False

def fast_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Tier 1 - Fast and memory-sensitive data cleaning.
    - Removes duplicate rows.
    - Standardizes numeric data types.
    - Applies simple median/mode imputation (or time-series interpolation).
    - Automatically parses and extracts datetime fields.
    Returns:
        tuple[pd.DataFrame, list[dict]]: Cleaned DataFrame and Audit Log.
    """
    audit_log = []
    if df.empty:
        return df.copy(), audit_log

    # Copy and reset index to ensure clean 0-based positioning matching frontend records
    df_clean = df.copy()
    df_clean.reset_index(drop=True, inplace=True)
    
    # 1. Deduplicate
    initial_len = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    rows_dropped = initial_len - len(df_clean)
    if rows_dropped > 0:
        audit_log.append({
            "column": "Dataset-wide",
            "issue": "Duplicate rows detected",
            "resolution": f"Removed {rows_dropped} duplicate rows"
        })
    
    # Check if this is a time-series dataset
    is_ts = _is_time_series(df_clean)
    
    # 2. Coerce string-encoded numbers to numeric (Maintain 80% threshold per Directive 4)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            non_nulls = df_clean[col].dropna()
            if not non_nulls.empty:
                numeric_conv = pd.to_numeric(non_nulls, errors='coerce')
                if numeric_conv.notnull().mean() > 0.8:
                    original_values = df_clean[col].copy()
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    changed_mask = original_values.notna() & df_clean[col].isna()
                    converted_mask = original_values.notna() & df_clean[col].notna() & (original_values.astype(str) != df_clean[col].astype(str))
                    total_changed_mask = changed_mask | converted_mask
                    
                    if total_changed_mask.any():
                        audit_log.append({
                            "column": col,
                            "issue": "String-encoded numeric format",
                            "resolution": f"Coerced {total_changed_mask.sum()} value(s) to float"
                        })
                    
    # 3. Imputation (Interpolation for Time-Series, else Median / Mode)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if is_ts:
        for col in numeric_cols:
            null_mask = df_clean[col].isna()
            if null_mask.any():
                interpolated = df_clean[col].interpolate(method='linear').ffill().bfill()
                audit_log.append({
                    "column": col,
                    "issue": f"{null_mask.sum()} Missing Value(s)",
                    "resolution": "Time-series linear interpolation"
                })
                df_clean[col] = interpolated
                
        for col in categorical_cols:
            null_mask = df_clean[col].isna()
            if null_mask.any():
                filled = df_clean[col].ffill().bfill()
                audit_log.append({
                    "column": col,
                    "issue": f"{null_mask.sum()} Missing Value(s)",
                    "resolution": "Time-series forward/backward fill"
                })
                df_clean[col] = filled
    else:
        for col in numeric_cols:
            null_mask = df_clean[col].isna()
            if null_mask.any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                else:
                    non_nulls = df_clean[col].dropna()
                    if len(non_nulls) > 0:
                        int_count = (non_nulls % 1 == 0).sum()
                        decimal_count = (non_nulls % 1 != 0).sum()
                        if int_count > decimal_count:
                            median_val = round(median_val)
                        else:
                            median_val = round(median_val, 2)
                audit_log.append({
                    "column": col,
                    "issue": f"{null_mask.sum()} Missing Value(s)",
                    "resolution": f"Simple median imputation (filled with {median_val})"
                })
                df_clean[col] = df_clean[col].fillna(median_val)
                
        for col in categorical_cols:
            null_mask = df_clean[col].isna()
            if null_mask.any():
                mode_vals = df_clean[col].mode()
                mode_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                audit_log.append({
                    "column": col,
                    "issue": f"{null_mask.sum()} Missing Value(s)",
                    "resolution": f"Simple mode imputation (filled with '{mode_val}')"
                })
                df_clean[col] = df_clean[col].fillna(mode_val)
 
    # 4. Auto-Datetime Extraction
    for col in categorical_cols:
        non_nulls = df_clean[col].dropna()
        if not non_nulls.empty and isinstance(non_nulls.iloc[0], str):
            try:
                sample_vals = non_nulls.head(50)
                first_val = str(sample_vals.iloc[0])
                if any(sep in first_val for sep in ['-', '/', ':', ' ']) or len(first_val) > 6:
                    parsed_sample = pd.to_datetime(sample_vals, errors='coerce')
                    if parsed_sample.notnull().mean() > 0.7:
                        dt_series = pd.to_datetime(df_clean[col], errors='coerce')
                        dt_series = dt_series.ffill().bfill()
                        
                        df_clean[f"{col}_year"] = dt_series.dt.year
                        df_clean[f"{col}_month"] = dt_series.dt.month
                        df_clean[f"{col}_day"] = dt_series.dt.day
                        df_clean[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                        
                        df_clean[col] = dt_series
                        audit_log.append({
                            "column": col,
                            "issue": "String-encoded date/time format",
                            "resolution": f"Parsed datetime and extracted features ({col}_year, {col}_month, {col}_day, {col}_is_weekend)"
                        })
            except Exception:
                pass
                
    return df_clean, audit_log

def deep_clean(df: pd.DataFrame, scaling: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """
    Tier 2 - Deep and comprehensive ML-driven data cleaning.
    - Inherits all Fast Clean features.
    - Applies Advanced Predictive Imputation (KNN/MICE) or Time-Series interpolation.
    - Performs fuzzy text clustering to merge typos (excluding short strings < 5, keys/IDs).
    - Detects multivariate outliers via Isolation Forest.
    - Optionally scales numeric features.
    Returns:
        tuple[pd.DataFrame, list[dict]]: Cleaned DataFrame and Audit Log.
    """
    if df.empty:
        return df.copy(), []

    # 1. Run Fast Clean first
    df_clean, audit_log = fast_clean(df)
    
    is_ts = _is_time_series(df_clean)
    
    # 2. Run Imputation
    if is_ts:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            null_mask = df_clean[col].isna()
            if null_mask.any():
                interpolated = df_clean[col].interpolate(method='linear').ffill().bfill()
                df_clean[col] = interpolated
    else:
        # Downgrade Strategy for Big Datasets
        # If dataset exceeds 1,000,000 rows and missingness > 20% in a column, bypass predictive imputation
        bypassed_cols = []
        n_rows = len(df_clean)
        if n_rows > 1000000:
            for col in df_clean.columns:
                missing_frac = df_clean[col].isna().mean()
                if missing_frac > 0.20:
                    bypassed_cols.append(col)
                    null_mask = df_clean[col].isna()
                    if null_mask.any():
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            median_val = df_clean[col].median()
                            if pd.isna(median_val):
                                median_val = 0
                            else:
                                non_nulls = df_clean[col].dropna()
                                if len(non_nulls) > 0:
                                    int_count = (non_nulls % 1 == 0).sum()
                                    decimal_count = (non_nulls % 1 != 0).sum()
                                    if int_count > decimal_count:
                                        median_val = round(median_val)
                                    else:
                                        median_val = round(median_val, 2)
                            df_clean[col] = df_clean[col].fillna(median_val)
                            res_msg = f"Simple median imputation fallback (filled with {median_val}) to prevent CPU exhaustion"
                        else:
                            mode_vals = df_clean[col].mode()
                            mode_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                            df_clean[col] = df_clean[col].fillna(mode_val)
                            res_msg = f"Simple mode imputation fallback (filled with '{mode_val}') to prevent CPU exhaustion"
                        
                        audit_log.append({
                            "column": col,
                            "issue": "High missingness in large data (>20% missing values in >1M rows)",
                            "resolution": res_msg
                        })

        df_before_impute = df_clean.copy()
        df_clean = advanced_impute(df_clean, method="auto")
        
        # Log difference after advanced imputation
        for col in df_clean.columns:
            if col in df_before_impute.columns and col not in bypassed_cols:
                null_mask = df_before_impute[col].isna() & df_clean[col].notna()
                if null_mask.any():
                    audit_log.append({
                        "column": col,
                        "issue": f"{null_mask.sum()} Missing Value(s)",
                        "resolution": "Advanced predictive imputation (KNN/MICE)"
                    })
                        
    # 3. Fuzzy Text Matching & Consolidation with Safety Exclusions
    categorical_cols = df_clean.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in categorical_cols:
        non_nulls = df_clean[col].dropna()
        if non_nulls.empty:
            continue
            
        avg_len = non_nulls.astype(str).str.len().mean()
        if avg_len < 5:
            continue
            
        if _is_id_or_key_column(col, df_clean[col]):
            continue
            
        unique_vals = [str(x) for x in non_nulls.unique() if len(str(x).strip()) > 1]
        if len(unique_vals) < 2 or len(unique_vals) > 100:
            continue
            
        value_counts = df_clean[col].value_counts().to_dict()
        mapping = {}
        processed = set()
        
        for val in sorted(unique_vals, key=len, reverse=True):
            if val in processed:
                continue
                
            matches = difflib.get_close_matches(val, unique_vals, n=5, cutoff=0.85)
            if len(matches) > 1:
                dominant = max(matches, key=lambda x: value_counts.get(x, 0))
                for match in matches:
                    if match != dominant:
                        mapping[match] = dominant
                        processed.add(match)
                        
        if mapping:
            # Count matches
            match_mask = df_clean[col].isin(mapping.keys())
            match_count = match_mask.sum()
            if match_count > 0:
                audit_log.append({
                    "column": col,
                    "issue": f"{match_count} spelling variation(s) / typo(s) detected",
                    "resolution": "Fuzzy consolidated string values using difflib"
                })
            df_clean[col] = df_clean[col].astype(str).replace(mapping)
 
    # 4. Multivariate Outlier Detection (Isolation Forest)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cols_for_iforest = [
        c for c in numeric_cols 
        if not c.endswith('_year') and not c.endswith('_month') and not c.endswith('_day') and not c.endswith('_is_weekend') and c != 'is_anomaly'
    ]
    
    if len(cols_for_iforest) >= 2:
        try:
            n_rows = len(df_clean)
            if n_rows > 100000:
                # Downsample train data to prevent memory issues on 8GB RAM VPS
                train_sample = df_clean[cols_for_iforest].sample(n=100000, random_state=42)
            else:
                train_sample = df_clean[cols_for_iforest]

            scaler = RobustScaler()
            scaled_train = scaler.fit_transform(train_sample)
            
            clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            clf.fit(scaled_train)

            # Score in streaming chunks to keep memory usage under 4-5GB
            chunk_size = 100000
            preds = []
            for i in range(0, n_rows, chunk_size):
                chunk = df_clean[cols_for_iforest].iloc[i : i + chunk_size]
                scaled_chunk = scaler.transform(chunk)
                preds.extend(clf.predict(scaled_chunk))
            
            df_clean["is_anomaly"] = np.where(np.array(preds) == -1, 1, 0)
            
            anomalies = df_clean[df_clean["is_anomaly"] == 1]
            if not anomalies.empty:
                audit_log.append({
                    "column": "Dataset-wide",
                    "issue": "Multivariate Outliers detected",
                    "resolution": f"Flagged {len(anomalies)} anomalous row(s) via Isolation Forest (contamination=0.05)"
                })
        except Exception as e:
            print(f"[Deep Clean] Outlier detection failed: {e}")
            df_clean["is_anomaly"] = 0
    else:
        df_clean["is_anomaly"] = 0
 
    # 5. Optional Feature Scaling
    if scaling and len(cols_for_iforest) > 0:
        try:
            scaler = RobustScaler()
            df_clean[cols_for_iforest] = scaler.fit_transform(df_clean[cols_for_iforest])
            audit_log.append({
                "column": ", ".join(cols_for_iforest),
                "issue": "Unscaled numeric features",
                "resolution": "Applied RobustScaler to normalize numeric distributions"
            })
        except Exception as e:
            print(f"[Deep Clean] Feature scaling failed: {e}")
            
    return df_clean, audit_log
