import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    total_rows, total_cols = df.shape
    col_types = {"numerical": [], "categorical": [], "datetime": []}
    col_details = {}
    for col in df.columns:
        is_dt = False
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            is_dt = True
        else:
            if df[col].dtype == "object":
                sample = df[col].dropna().head(10)
                if not sample.empty:
                    try:
                        parseable_dates = 0
                        for val in sample:
                            if isinstance(val, str) and re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', val):
                                parseable_dates += 1
                        if parseable_dates >= len(sample) * 0.8:
                            is_dt = True
                    except:
                        pass
        missing_count = int(df[col].isnull().sum())
        missing_pct = float(missing_count / total_rows) * 100
        unique_count = int(df[col].nunique())
        if is_dt:
            col_types["datetime"].append(col)
            inferred_type = "Datetime"
        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            if unique_count <= 10 and pd.api.types.is_integer_dtype(df[col]):
                col_types["categorical"].append(col)
                inferred_type = "Categorical (Low Cardinality Numeric)"
            else:
                col_types["numerical"].append(col)
                inferred_type = "Numerical"
        else:
            col_types["categorical"].append(col)
            inferred_type = "Categorical"
        col_details[col] = {
            "type": inferred_type,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "sample_values": df[col].dropna().head(5).tolist()
        }
    return {
        "shape": (total_rows, total_cols),
        "column_types": col_types,
        "column_details": col_details,
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
    }

def rank_target_candidates(df: pd.DataFrame, col_types: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    candidates = []
    total_rows = len(df)
    target_regex = re.compile(r'(target|label|class|y|status|output|sale_price|price|revenue|income|clicked|survived|churn|fraud|default)', re.IGNORECASE)
    id_regex = re.compile(r'(id|name|idx|uuid|guid|row|index|key|serial|code|phone|email)', re.IGNORECASE)
    for idx, col in enumerate(df.columns):
        score = 0.0
        reasons = []
        unique_vals = df[col].nunique()
        missing_vals = df[col].isnull().sum()
        is_id_name = bool(id_regex.search(col))
        is_pk = (unique_vals == total_rows)
        if is_pk:
            score -= 10.0
            reasons.append("Identified as unique identifier / primary key (100% unique values).")
        elif is_id_name and unique_vals > total_rows * 0.5:
            score -= 5.0
            reasons.append("Looks like an identifier column with high uniqueness.")
        if unique_vals <= 1:
            score -= 20.0
            reasons.append("Column has 1 or fewer unique values.")
        if missing_vals == total_rows:
            score -= 20.0
            reasons.append("Column is fully empty/null.")
        if target_regex.search(col):
            score += 8.0
            reasons.append(f"Name matches key target identifier ('{col}').")
        if idx == len(df.columns) - 1:
            score += 4.0
            reasons.append("Positioned as the last column of the dataset.")
        elif idx == 0:
            score -= 2.0
            reasons.append("Positioned as the first column (often an ID).")
        if col in col_types["categorical"]:
            if unique_vals == 2:
                score += 5.0
                reasons.append("Categorical column with exactly 2 classes (Binary Classification).")
            elif 2 < unique_vals <= 10:
                score += 4.0
                reasons.append(f"Categorical column with {unique_vals} classes (Multiclass Classification).")
            elif unique_vals > 50 and not is_id_name:
                score -= 3.0
                reasons.append("High cardinality categorical columns are rarely direct targets.")
        elif col in col_types["numerical"]:
            if not is_id_name and unique_vals > 10:
                score += 3.0
                reasons.append("Continuous numeric column (Regression target).")
        if missing_vals > 0:
            pct = (missing_vals / total_rows) * 100
            score -= (pct / 10.0)
            reasons.append(f"Has {pct:.1f}% missing values.")
        if not reasons:
            reasons.append("Standard feature column.")
        
        # Determine inferred type for the frontend
        if unique_vals == 2:
            inferred_type = "binary"
        elif col in col_types["categorical"] or unique_vals <= 10:
            inferred_type = "categorical"
        else:
            inferred_type = "continuous"

        candidates.append({
            "column_name": col,
            "column": col,
            "score": round(score, 2),
            "suitability_score": round(score, 2),
            "inferred_type": inferred_type,
            "reason": "; ".join(reasons)
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def infer_task_type(df: pd.DataFrame, target_col: str, col_types: Dict[str, List[str]]) -> str:
    unique_vals = df[target_col].nunique()
    if target_col in col_types["categorical"] or df[target_col].dtype == "object" or df[target_col].dtype == "bool":
        if unique_vals == 2:
            return "binary"
        else:
            return "multiclass"
    if unique_vals == 2:
        return "binary"
    elif unique_vals <= 10:
        return "multiclass"
    else:
        return "regression"
