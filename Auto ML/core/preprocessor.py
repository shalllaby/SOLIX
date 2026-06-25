import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any, Optional
import re

def sanitize_column_name(name: str) -> str:
    """
    Sanitizes a single column name into a safe ML-compatible name:
    - Removes special JSON characters (like {, }, :, ,, etc.)
    - Removes quotes and brackets (like ", ', [, ], (, ))
    - Replaces spaces and other non-alphanumeric symbols with underscores
    - Preserves uniqueness
    """
    name_str = str(name)
    cleaned = re.sub(r'[\'"\[\]\(\)\{\}]', '', name_str)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"feat_{cleaned}" if cleaned else "feat"
    return cleaned

def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes all column names of the DataFrame to be ML-compatible,
    ensuring uniqueness by appending counter suffixes if duplicates arise.
    """
    new_cols = []
    seen = {}
    for col in df.columns:
        san = sanitize_column_name(col)
        if san in seen:
            seen[san] += 1
            san_unique = f"{san}_{seen[san]}"
        else:
            seen[san] = 0
            san_unique = san
        new_cols.append(san_unique)
        
    df_copy = df.copy()
    df_copy.columns = new_cols
    return df_copy

def coerce_hidden_numericals(
    df: pd.DataFrame,
    force_numeric: List[str] = None,
    conversion_threshold: float = 0.70,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detects and casts 'hidden numerical' columns — object-dtype columns that
    actually contain numeric data corrupted by blank spaces or minor parsing
    artifacts (e.g., TotalCharges in the Telco Customer Churn dataset).

    Strategy:
      1. Strip leading/trailing whitespace from string values.
      2. Attempt pd.to_numeric(errors='coerce') on the stripped column.
      3. Accept the cast if >= conversion_threshold of non-null values
         convert successfully. The resulting NaNs from blanks/parse errors
         are handled downstream by SimpleImputer(strategy='median').

    Columns listed in `force_numeric` are cast unconditionally, bypassing
    the threshold check — these come from the LLM profiler's suggestion.

    Args:
        df: Input DataFrame. A copy is made; the original is not modified.
        force_numeric: Column names explicitly forced to numeric (e.g., from
                       the Agentic Profiler's `force_numeric` output field).
        conversion_threshold: Minimum fraction of non-null values that must
                              convert to float to auto-accept the cast.
                              Hardcoded at 0.70 — not exposed to the UI.

    Returns:
        Tuple of (coerced_df, coerced_cols) where coerced_cols is the list
        of column names that were successfully cast to numeric.
    """
    df = df.copy()
    coerced_cols = []
    force_set = set(force_numeric or [])

    for col in df.columns:
        if df[col].dtype != object:
            continue  # Already numeric or datetime — nothing to do

        # Priority: force-coerce columns explicitly requested by the LLM profiler
        if col in force_set:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip(), errors="coerce"
            )
            coerced_cols.append(col)
            continue

        # Auto-detect path: strip whitespace, attempt numeric cast
        stripped = df[col].astype(str).str.strip()
        attempted = pd.to_numeric(stripped, errors="coerce")

        non_null_mask = df[col].notna()
        n_non_null = int(non_null_mask.sum())
        if n_non_null == 0:
            continue  # Fully empty column — skip

        # Accept cast only if enough values converted successfully
        n_converted = int(attempted[non_null_mask].notna().sum())
        conversion_rate = n_converted / n_non_null

        if conversion_rate >= conversion_threshold:
            df[col] = attempted
            coerced_cols.append(col)

    return df, coerced_cols


def build_preprocessing_pipeline(
    numerical_cols: List[str],
    low_card_cols: List[str],
    high_card_cols: List[str],
    scaling_method: str = "standard",   # standard, minmax, robust, none
    encoding_method: str = "onehot",    # onehot, ordinal, none
) -> ColumnTransformer:
    """
    Creates and returns a Scikit-Learn ColumnTransformer pipeline 
    configured for numerical, low-cardinality, and high-cardinality categorical features.
    """
    transformers = []
    
    # 1. Numerical Pipeline
    if numerical_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        
        if scaling_method == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaling_method == "robust":
            num_steps.append(("scaler", RobustScaler()))
            
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, numerical_cols))
        
    # 2. Low-Cardinality Categorical Pipeline (One-Hot / Ordinal)
    if low_card_cols:
        cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        
        if encoding_method == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif encoding_method == "ordinal":
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
            
        low_card_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("low_card", low_card_pipeline, low_card_cols))

    # 3. High-Cardinality Categorical Pipeline (Target Encoding / Ordinal)
    if high_card_cols:
        cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        
        if encoding_method == "onehot":
            try:
                from sklearn.preprocessing import TargetEncoder
                # TargetEncoder cv=5 defaults, random_state for consistency
                cat_steps.append(("encoder", TargetEncoder(cv=5, random_state=42)))
            except ImportError:
                # Fallback to OrdinalEncoder if scikit-learn version is old (<1.3)
                cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        elif encoding_method == "ordinal":
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
            
        high_card_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("high_card", high_card_pipeline, high_card_cols))
        
    # Combine
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop" # Drop datetime or ID columns that shouldn't be trained on
    )
    
    return preprocessor

def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    col_types: Dict[str, List[str]],
    task_type: str,
    test_size: float = 0.2,
    scaling_method: str = "standard",
    encoding_method: str = "onehot",
    blacklist: list = None,
    force_numeric: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, ColumnTransformer, Optional[LabelEncoder]]:
    """
    Splits features and target, processes target, fits the preprocessor 
    on the train set, transforms train/test sets, and returns everything.
    Output features are strictly returned as pandas DataFrames with preserved names.
    """
    # 0. Global Feature-name Sanitization before training
    df = sanitize_features(df)
    target_col = sanitize_column_name(target_col)
    col_types = {
        k: [sanitize_column_name(c) for c in v]
        for k, v in col_types.items()
    }

    # 0.1 Numeric Rescue — Detect and cast hidden numerical columns BEFORE
    # pipeline routing. This fixes the 'TotalCharges trap': columns stored as
    # object/string due to blank spaces that would otherwise be misrouted to
    # TargetEncoder, causing catastrophic tree overfitting.
    # LLM profiler's force_numeric list acts as a secondary override on top.
    force_numeric_sanitized = [
        sanitize_column_name(c) for c in (force_numeric or [])
    ]
    df, coerced_cols = coerce_hidden_numericals(
        df, force_numeric=force_numeric_sanitized
    )
    if coerced_cols:
        # Re-route coerced columns: remove from categorical, add to numerical.
        # This ensures the downstream pipeline applies median imputation +
        # numeric scaling instead of TargetEncoder.
        coerced_set = set(coerced_cols)
        col_types["numerical"] = list(
            dict.fromkeys(col_types.get("numerical", []) + coerced_cols)
        )
        for cat_key in ("categorical",):
            col_types[cat_key] = [
                c for c in col_types.get(cat_key, []) if c not in coerced_set
            ]

    # 0.5 Dynamic Blacklist — Drop AI-flagged or user-specified columns
    if blacklist:
        sanitized_blacklist = {sanitize_column_name(c) for c in blacklist}
        bl_found = [c for c in df.columns if c in sanitized_blacklist and c != target_col]
        if bl_found:
            df = df.drop(columns=bl_found)
        # Remove blacklisted columns from col_types to keep feature lists consistent
        for k in col_types:
            col_types[k] = [c for c in col_types[k] if c not in sanitized_blacklist]

    # Exclude target from feature lists
    num_features = [col for col in col_types.get("numerical", []) if col != target_col]
    cat_features = [col for col in col_types.get("categorical", []) if col != target_col]
    
    # 1. Zero-variance and ID detection logic
    cols_to_drop = []
    total_rows = len(df)
    
    for col in num_features + cat_features:
        if col not in df.columns:
            continue
        col_data = df[col]
        n_unique = col_data.nunique(dropna=True)
        
        # Zero-variance check
        if n_unique <= 1:
            cols_to_drop.append(col)
            continue
            
        # Near-zero variance (99.5% identical)
        val_counts = col_data.value_counts(normalize=True, dropna=True)
        if not val_counts.empty and val_counts.iloc[0] > 0.995:
            cols_to_drop.append(col)
            continue
            
        # ID-like checks
        is_id_name = bool(re.search(r'(?i)(id|uuid|key|code|index|hash)', col))
        uniqueness_ratio = n_unique / total_rows
        
        if uniqueness_ratio > 0.90:
            if col in cat_features or is_id_name:
                cols_to_drop.append(col)
                continue
        elif is_id_name and uniqueness_ratio > 0.50:
            cols_to_drop.append(col)
            continue
            
    # Filter final features
    num_features = [c for c in num_features if c not in cols_to_drop]
    cat_features = [c for c in cat_features if c not in cols_to_drop]
    
    # Differentiate categorical features by cardinality threshold (<= 15 vs > 15)
    low_card_features = []
    high_card_features = []
    for col in cat_features:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 15:
            low_card_features.append(col)
        else:
            high_card_features.append(col)

    X = df[num_features + low_card_features + high_card_features]
    y = df[target_col]
    
    # Target Encoding
    target_encoder = None
    if task_type in ["binary", "multiclass"]:
        target_encoder = LabelEncoder()
        y_clean = y.fillna(y.mode()[0] if not y.mode().empty else "missing")
        y_processed = target_encoder.fit_transform(y_clean)
    else:
        y_processed = y.fillna(y.median()).values
        
    # Train/Test Split
    stratify_y = y_processed if task_type in ["binary", "multiclass"] else None
    
    if stratify_y is not None:
        unique_counts = pd.Series(stratify_y).value_counts()
        if (unique_counts < 2).any():
            stratify_y = None
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_processed, test_size=test_size, random_state=42, stratify=stratify_y
    )
    
    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numerical_cols=num_features,
        low_card_cols=low_card_features,
        high_card_cols=high_card_features,
        scaling_method=scaling_method,
        encoding_method=encoding_method
    )
    
    # Try setting modern pandas output for Scikit-Learn
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass
    
    # Fit-transform train set (requires y_train for TargetEncoder), transform test set
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Force clean column names and convert to pandas DataFrames
    raw_feature_names = get_processed_feature_names(preprocessor)
    feature_names = []
    seen = {}
    for col in raw_feature_names:
        san = sanitize_column_name(col)
        if san in seen:
            seen[san] += 1
            san_unique = f"{san}_{seen[san]}"
        else:
            seen[san] = 0
            san_unique = san
        feature_names.append(san_unique)
    
    if isinstance(X_train_processed, pd.DataFrame):
        X_train_processed.columns = feature_names
        X_test_processed.columns = feature_names
    else:
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, target_encoder

def get_processed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Utility to retrieve feature names outputted by the fitted ColumnTransformer.
    Important for feature importance graphing!
    """
    feature_names = []
    
    try:
        for name, pipe, cols in preprocessor.transformers_:
            if name == "remainder" or pipe == "drop":
                continue
            step_names = [s[0] for s in pipe.steps]
            
            if "encoder" in step_names:
                encoder_idx = step_names.index("encoder")
                encoder = pipe.steps[encoder_idx][1]
                if hasattr(encoder, "get_feature_names_out"):
                    names = encoder.get_feature_names_out(cols)
                    feature_names.extend(names)
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
    except Exception:
        total_cols = preprocessor.n_features_in_ if hasattr(preprocessor, 'n_features_in_') else 10
        feature_names = [f"Feature_{i}" for i in range(total_cols)]
        
    return feature_names
