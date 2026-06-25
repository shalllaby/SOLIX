import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs lightweight profiling of the dataset.
    Detects numerical, categorical, and datetime columns.
    Returns general statistics and metadata.
    """
    total_rows, total_cols = df.shape
    
    col_types = {
        "numerical": [],
        "categorical": [],
        "datetime": []
    }
    
    col_details = {}
    
    for col in df.columns:
        # Check datetime
        is_dt = False
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            is_dt = True
        else:
            # Try lightweight parse for string columns
            if df[col].dtype == "object":
                # Sample non-null values
                sample = df[col].dropna().head(10)
                if not sample.empty:
                    try:
                        # Check if matches standard date patterns
                        # e.g., YYYY-MM-DD or DD/MM/YYYY
                        parseable_dates = 0
                        for val in sample:
                            if isinstance(val, str) and re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', val):
                                parseable_dates += 1
                        if parseable_dates >= len(sample) * 0.8:
                            is_dt = True
                    except:
                        pass
        
        # Missing count
        missing_count = int(df[col].isnull().sum())
        missing_pct = float(missing_count / total_rows) * 100
        unique_count = int(df[col].nunique())
        
        if is_dt:
            col_types["datetime"].append(col)
            inferred_type = "Datetime"
        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            # If numeric but very low cardinality (e.g. binary 0/1 or tiny integer class), could be categorical, 
            # but standard ML prep treats high-cardinality ints as numerical and low as numerical or categorical.
            # We classify it as numerical, and preprocessor will handle it.
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
    """
    Ranks the likely target columns of the dataset.
    Returns a sorted list of dictionaries: [ { "column": name, "score": float, "reason": str } ]
    """
    candidates = []
    total_rows = len(df)
    
    # Common target keyword regexes
    target_regex = re.compile(r'(target|label|class|y|status|output|sale_price|price|revenue|income|clicked|survived|churn|fraud|default)', re.IGNORECASE)
    id_regex = re.compile(r'(id|name|idx|uuid|guid|row|index|key|serial|code|phone|email)', re.IGNORECASE)
    
    for idx, col in enumerate(df.columns):
        score = 0.0
        reasons = []
        
        unique_vals = df[col].nunique()
        missing_vals = df[col].isnull().sum()
        
        # Rule 1: ID columns make horrible targets
        is_id_name = bool(id_regex.search(col))
        is_pk = (unique_vals == total_rows)
        
        if is_pk:
            # Unique ID columns are terrible targets
            score -= 10.0
            reasons.append("Identified as unique identifier / primary key (100% unique values).")
        elif is_id_name and unique_vals > total_rows * 0.5:
            score -= 5.0
            reasons.append("Looks like an identifier column with high uniqueness.")
            
        # Rule 2: Constant or fully null columns cannot be targets
        if unique_vals <= 1:
            score -= 20.0
            reasons.append("Column has 1 or fewer unique values.")
        if missing_vals == total_rows:
            score -= 20.0
            reasons.append("Column is fully empty/null.")
            
        # Rule 3: Exact matches to target keywords
        if target_regex.search(col):
            score += 8.0
            reasons.append(f"Name matches key target identifier ('{col}').")
            
        # Rule 4: Column position (targets are usually the last column)
        if idx == len(df.columns) - 1:
            score += 4.0
            reasons.append("Positioned as the last column of the dataset.")
        elif idx == 0:
            score -= 2.0
            reasons.append("Positioned as the first column (often an ID).")
            
        # Rule 5: Target cardinality & types
        # Categorical columns with 2-10 unique values are great for classification
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
        # Numerical columns that aren't IDs can represent regression targets
        elif col in col_types["numerical"]:
            if not is_id_name and unique_vals > 10:
                score += 3.0
                reasons.append("Continuous numeric column (Regression target).")
                
        # Rule 6: Missing values penalty
        if missing_vals > 0:
            pct = (missing_vals / total_rows) * 100
            score -= (pct / 10.0) # deduct up to 10 points
            reasons.append(f"Has {pct:.1f}% missing values (targets shouldn't require heavy imputation).")
            
        # Compile natural language reason
        if not reasons:
            reasons.append("Standard feature column.")
            
        candidates.append({
            "column": col,
            "score": round(score, 2),
            "reason_en": "; ".join(reasons),
            "reason_ar": translate_reasons_to_ar(col, unique_vals, is_pk, is_id_name, idx, len(df.columns), col in col_types["categorical"])
        })
        
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def translate_reasons_to_ar(col: str, unique_vals: int, is_pk: bool, is_id_name: bool, idx: int, num_cols: int, is_cat: bool) -> str:
    """
    Translates targeting reasons to high-quality Arabic text.
    """
    reasons = []
    
    # Standard keywords translation
    target_keywords = ['target', 'label', 'class', 'y', 'status', 'output', 'sale_price', 'price', 'revenue', 'income', 'clicked', 'survived', 'churn', 'fraud', 'default']
    matches_target = any(kw in col.lower() for kw in target_keywords)
    
    if is_pk:
        reasons.append("تم تحديده كمعرف فريد / مفتاح أساسي (القيم فريدة بنسبة 100%).")
    elif is_id_name and unique_vals > 10:
        reasons.append("يبدو كعمود معرف (ID) بسبب نسبة فرادة القيم المرتفعة.")
        
    if unique_vals <= 1:
        reasons.append("العمود يحتوي على قيمة واحدة فريدة أو أقل.")
        
    if matches_target:
        reasons.append(f"الاسم يتطابق مع الكلمات الدلالية الشائعة للأهداف ('{col}').")
        
    if idx == num_cols - 1:
        reasons.append("يقع في نهاية الجدول (الترتيب الشائع للأعمدة المستهدفة).")
    elif idx == 0:
        reasons.append("يقع في بداية الجدول (الترتيب الشائع للأعمدة المعرفة).")
        
    if is_cat:
        if unique_vals == 2:
            reasons.append("عمود فئوي يحتوي على فئتين بالضبط (تصنيف ثنائي).")
        elif 2 < unique_vals <= 10:
            reasons.append(f"عمود فئوي يحتوي على {unique_vals} فئات (تصنيف متعدد الفئات).")
    else:
        if not is_id_name and unique_vals > 10:
            reasons.append("عمود رقمي مستمر (هدف انحدار).")
            
    if not reasons:
        reasons.append("عمود بيانات اعتيادي.")
        
    return "؛ ".join(reasons)

def infer_task_type(df: pd.DataFrame, target_col: str, col_types: Dict[str, List[str]]) -> str:
    """
    Infers the ML task type based on target column characteristics.
    Returns: 'binary', 'multiclass', or 'regression'
    """
    unique_vals = df[target_col].nunique()
    
    # If categorical, it is classification
    if target_col in col_types["categorical"] or df[target_col].dtype == "object" or df[target_col].dtype == "bool":
        if unique_vals == 2:
            return "binary"
        else:
            return "multiclass"
            
    # If numerical, check cardinality
    if unique_vals == 2:
        return "binary"
    elif unique_vals <= 10:
        return "multiclass"
    else:
        return "regression"
