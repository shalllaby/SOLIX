"""
core/copilot/imputer.py
=======================
Advanced predictive imputation helper using scikit-learn (KNNImputer & IterativeImputer).
Provides seamless categorical encoding/decoding and automated decision heuristics.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

def advanced_impute(df: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    """
    Apply advanced predictive imputation (KNNImputer or IterativeImputer) to the dataframe.
    Automatically handles categorical columns by encoding/decoding them, and falls
    back to simple median/mode imputation if datasets are small or have low missingness.

    Parameters:
        df (pd.DataFrame): The input dataframe with missing values.
        method (str): "auto", "simple", "knn", or "iterative".

    Returns:
        pd.DataFrame: A fully imputed copy of the dataframe.
    """
    if df.empty:
        return df.copy()

    df_clean = df.copy()
    
    # Separate numeric, categorical, and other columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    # Datetime or other columns that shouldn't be processed by scikit-learn imputers
    other_cols = [c for c in df_clean.columns if c not in numeric_cols and c not in categorical_cols]

    # Handle entirely empty columns first (they cannot be imputed by ML, so we drop or constant-fill them)
    entirely_empty_cols = [col for col in df_clean.columns if df_clean[col].isnull().all()]
    for col in entirely_empty_cols:
        if col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna("Unknown")

    # Re-evaluate column categories after cleaning entirely empty columns
    numeric_cols = [c for c in numeric_cols if c not in entirely_empty_cols]
    categorical_cols = [c for c in categorical_cols if c not in entirely_empty_cols]

    # Handle temporal/other columns with simple propagation (ffill/bfill)
    for col in other_cols:
        df_clean[col] = df_clean[col].ffill().bfill()
        # If still has NaNs (e.g. all empty or single row)
        if df_clean[col].isnull().any():
            # Use mode or string fallback
            mode_vals = df_clean[col].mode()
            df_clean[col] = df_clean[col].fillna(mode_vals[0] if not mode_vals.empty else pd.NaT)

    # 1. Determine Imputation Method
    chosen_method = method
    if method == "auto":
        chosen_method = _choose_imputation_method(df_clean, numeric_cols)

    # If chosen method is simple, perform standard median/mode imputation and return
    if chosen_method == "simple":
        for col in numeric_cols:
            median_val = df_clean[col].median()
            # If median is NaN (e.g. all NaNs, though handled above), fallback to 0
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
        for col in categorical_cols:
            mode_vals = df_clean[col].mode()
            mode_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)
        return df_clean

    # 2. Encode Categorical Variables (preserving NaNs as np.nan)
    encoder_map = {}
    encoded_df = df_clean.copy()
    
    for col in categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        non_null_mask = df_clean[col].notnull()
        if non_null_mask.any():
            try:
                # Fit encoder on the non-null elements
                encoded_values = encoder.fit_transform(df_clean.loc[non_null_mask, [col]])
                # Map back to the dataframe
                encoded_df.loc[non_null_mask, col] = encoded_values.flatten()
                encoder_map[col] = encoder
            except Exception as e:
                print(f"[Advanced Impute] Warning: Failed to encode categorical column {col}: {e}")
                # Fallback: simple mode imputation for this column
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                encoded_df[col] = encoded_df[col].fillna(mode_val)
                
        # Coerce to float so it supports NaNs
        encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')

    # Drop columns that are completely non-numeric or couldn't be encoded (to prevent imputer crash)
    cols_to_impute = [col for col in encoded_df.columns if col not in other_cols]
    if not cols_to_impute:
        return df_clean

    # 3. Fit and Apply Scikit-learn Imputer
    try:
        if chosen_method == "knn":
            imputer = KNNImputer(n_neighbors=min(5, max(1, len(encoded_df) - 1)))
        else:  # iterative / MICE
            imputer = IterativeImputer(max_iter=10, random_state=42)

        imputed_array = imputer.fit_transform(encoded_df[cols_to_impute])
        imputed_subset = pd.DataFrame(imputed_array, columns=cols_to_impute, index=encoded_df.index)
        
        # Merge imputed data back
        for col in cols_to_impute:
            encoded_df[col] = imputed_subset[col]
            
    except Exception as e:
        print(f"[Advanced Impute] Error in predictive imputation ({chosen_method}): {e}. Falling back to simple imputation.")
        # Fallback to simple imputation on the original data
        return advanced_impute(df, method="simple")

    # 4. Decode Categorical Columns
    for col in categorical_cols:
        if col in encoder_map:
            # Round prediction to nearest category code
            encoded_df[col] = encoded_df[col].round().astype(int)
            # Clip within the category indices range
            max_idx = len(encoder_map[col].categories_[0]) - 1
            encoded_df[col] = encoded_df[col].clip(0, max_idx)
            
            # Decode using inverse_transform
            decoded = encoder_map[col].inverse_transform(encoded_df[[col]])
            df_clean[col] = decoded.flatten()
        else:
            # Fallback if encoding maps failed
            mode_vals = df_clean[col].mode()
            df_clean[col] = df_clean[col].fillna(mode_vals[0] if not mode_vals.empty else "Unknown")

    # 5. Restore Numerical Column Values & Original Types
    for col in numeric_cols:
        df_clean[col] = encoded_df[col]
        # Preserve original integer or float subtypes
        orig_dtype = df[col].dtype
        try:
            non_nulls = df[col].dropna()
            if len(non_nulls) > 0:
                int_count = (non_nulls % 1 == 0).sum()
                decimal_count = (non_nulls % 1 != 0).sum()
                if int_count > decimal_count:
                    df_clean[col] = df_clean[col].round()
                else:
                    df_clean[col] = df_clean[col].round(2)

            if pd.api.types.is_integer_dtype(orig_dtype):
                df_clean[col] = df_clean[col].round().astype(orig_dtype)
            else:
                df_clean[col] = df_clean[col].astype(orig_dtype)
        except Exception:
            pass

    return df_clean

def _choose_imputation_method(df: pd.DataFrame, numeric_cols: list) -> str:
    """Heuristic rule system to select the optimal imputation strategy."""
    n_rows, n_cols = df.shape
    total_missing_ratio = df.isnull().mean().mean()

    # Rule 1: Too small for ML
    if n_rows < 50 or n_cols < 2:
        return "simple"

    # Rule 1b: Too large for ML imputation (limit to 100k rows)
    if n_rows > 100000:
        return "simple"

    # Rule 2: Low missingness (<1% total missing) -> simple is faster and sufficient
    if total_missing_ratio < 0.01:
        return "simple"

    # Rule 3: Check correlations between numeric features
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            # Select upper triangle of correlation matrix
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            max_corr = upper_tri.max().max()
            
            # High correlation -> IterativeImputer (MICE) is best
            if not pd.isna(max_corr) and max_corr > 0.4:
                return "iterative"
        except Exception:
            pass

    # Default to KNN
    return "knn"
