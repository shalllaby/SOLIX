import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any, Optional
import re


def sanitize_column_name(name: str) -> str:
    name_str = str(name)
    cleaned = re.sub(r'[\'"\[\]\(\)\{\}]', '', name_str)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"feat_{cleaned}" if cleaned else "feat"
    return cleaned


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    coerced_cols = []
    force_set = set(force_numeric or [])
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if col in force_set:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            coerced_cols.append(col)
            continue
        stripped = df[col].astype(str).str.strip()
        attempted = pd.to_numeric(stripped, errors="coerce")
        non_null_mask = df[col].notna()
        n_non_null = int(non_null_mask.sum())
        if n_non_null == 0:
            continue
        n_converted = int(attempted[non_null_mask].notna().sum())
        if n_converted / n_non_null >= conversion_threshold:
            df[col] = attempted
            coerced_cols.append(col)
    return df, coerced_cols


def build_preprocessing_pipeline(
    numerical_cols, low_card_cols, high_card_cols,
    scaling_method="standard", encoding_method="onehot"
) -> ColumnTransformer:
    transformers = []
    if numerical_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scaling_method == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaling_method == "robust":
            num_steps.append(("scaler", RobustScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), numerical_cols))
    if low_card_cols:
        cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        if encoding_method == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        else:
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        transformers.append(("low_card", Pipeline(steps=cat_steps), low_card_cols))
    if high_card_cols:
        hc_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        if encoding_method == "onehot":
            try:
                from sklearn.preprocessing import TargetEncoder
                hc_steps.append(("encoder", TargetEncoder(cv=5, random_state=42)))
            except ImportError:
                hc_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            hc_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        transformers.append(("high_card", Pipeline(steps=hc_steps), high_card_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_data(
    df, target_col, col_types, task_type,
    test_size=0.2, scaling_method="standard",
    encoding_method="onehot", blacklist=None, force_numeric=None
):
    df = sanitize_features(df)
    target_col = sanitize_column_name(target_col)
    col_types = {k: [sanitize_column_name(c) for c in v] for k, v in col_types.items()}

    force_numeric_sanitized = [sanitize_column_name(c) for c in (force_numeric or [])]
    df, coerced_cols = coerce_hidden_numericals(df, force_numeric=force_numeric_sanitized)
    if coerced_cols:
        coerced_set = set(coerced_cols)
        col_types["numerical"] = list(dict.fromkeys(col_types.get("numerical", []) + coerced_cols))
        col_types["categorical"] = [c for c in col_types.get("categorical", []) if c not in coerced_set]

    if blacklist:
        sanitized_blacklist = {sanitize_column_name(c) for c in blacklist}
        bl_found = [c for c in df.columns if c in sanitized_blacklist and c != target_col]
        if bl_found:
            df = df.drop(columns=bl_found)
        for k in col_types:
            col_types[k] = [c for c in col_types[k] if c not in sanitized_blacklist]

    num_features = [c for c in col_types.get("numerical", []) if c != target_col]
    cat_features = [c for c in col_types.get("categorical", []) if c != target_col]

    cols_to_drop = []
    total_rows = len(df)
    for col in num_features + cat_features:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            cols_to_drop.append(col); continue
        vc = df[col].value_counts(normalize=True, dropna=True)
        if not vc.empty and vc.iloc[0] > 0.995:
            cols_to_drop.append(col); continue
        is_id = bool(re.search(r'(?i)(id|uuid|key|code|index|hash)', col))
        ur = n_unique / total_rows
        if ur > 0.90 and (col in cat_features or is_id):
            cols_to_drop.append(col); continue
        if is_id and ur > 0.50:
            cols_to_drop.append(col); continue

    num_features = [c for c in num_features if c not in cols_to_drop]
    cat_features = [c for c in cat_features if c not in cols_to_drop]

    low_card = [c for c in cat_features if df[c].nunique(dropna=True) <= 15]
    high_card = [c for c in cat_features if df[c].nunique(dropna=True) > 15]

    X = df[num_features + low_card + high_card]
    y = df[target_col]

    target_encoder = None
    if task_type in ["binary", "multiclass"]:
        target_encoder = LabelEncoder()
        y_clean = y.fillna(y.mode()[0] if not y.mode().empty else "missing")
        y_processed = target_encoder.fit_transform(y_clean)
    else:
        y_processed = y.fillna(y.median()).values

    stratify_y = y_processed if task_type in ["binary", "multiclass"] else None
    if stratify_y is not None:
        if (pd.Series(stratify_y).value_counts() < 2).any():
            stratify_y = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_processed, test_size=test_size, random_state=42, stratify=stratify_y
    )

    preprocessor = build_preprocessing_pipeline(num_features, low_card, high_card, scaling_method, encoding_method)
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

    X_train_p = preprocessor.fit_transform(X_train, y_train)
    X_test_p = preprocessor.transform(X_test)

    raw_names = get_processed_feature_names(preprocessor)
    feat_names = []
    seen = {}
    for col in raw_names:
        san = sanitize_column_name(col)
        if san in seen:
            seen[san] += 1
            feat_names.append(f"{san}_{seen[san]}")
        else:
            seen[san] = 0
            feat_names.append(san)

    if isinstance(X_train_p, pd.DataFrame):
        X_train_p.columns = feat_names
        X_test_p.columns = feat_names
    else:
        X_train_p = pd.DataFrame(X_train_p, columns=feat_names, index=X_train.index)
        X_test_p = pd.DataFrame(X_test_p, columns=feat_names, index=X_test.index)

    return X_train_p, X_test_p, y_train, y_test, preprocessor, target_encoder


def get_processed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names = []
    try:
        for name, pipe, cols in preprocessor.transformers_:
            if name == "remainder" or pipe == "drop":
                continue
            step_names = [s[0] for s in pipe.steps]
            if "encoder" in step_names:
                enc = pipe.steps[step_names.index("encoder")][1]
                if hasattr(enc, "get_feature_names_out"):
                    feature_names.extend(enc.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
    except Exception:
        n = preprocessor.n_features_in_ if hasattr(preprocessor, 'n_features_in_') else 10
        feature_names = [f"Feature_{i}" for i in range(n)]
    return feature_names
