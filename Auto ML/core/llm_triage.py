"""
core/llm_triage.py — SOL Agentic Model Router / LLM Triage
Groq-powered algorithm selection committee based on dataset characteristics.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger("SOL.ModelTriage")

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MAX_TOKENS = 1536
GROQ_TEMPERATURE = 0.0  # Zero temperature for deterministic routing results

SYSTEM_PROMPT = """You are the Lead Algorithm Selection Committee for the "Al-Dalil" Automated Machine Learning (AutoML) platform.
Your task is to analyze dataset metadata and select a subset of the 10 available base models that are most suited for this dataset, rejecting models that are mathematically redundant, computationally prohibitive, or highly prone to severe overfitting on this profile.

Candidates: [LogisticRegression, KNN, SVM, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost]

Rules for Model Selection:
1. KNN: Reject if columns > 50 or rows > 20,000 (curse of dimensionality / memory complexity scaling).
2. SVM: Reject if rows > 15,000 (O(N^3) complexity of SVM is too slow).
3. DecisionTree: Reject if task is complex (columns > 15) to prevent baseline overfitting.
4. Linear Models (LogisticRegression): Approve if dataset is classification to serve as baseline, or if dataset is small.
5. Tree Ensembles (RandomForest, ExtraTrees): Approve if tabular dataset requires non-linear interactions, but reject one of them if training budget/time is highly constrained (e.g. rows > 50,000).
6. Gradient Boosting (XGBoost, LightGBM, CatBoost): 
   - XGBoost: Approve if missing values exist or if high tabular accuracy is needed.
   - LightGBM: Approve if dataset has high categorical counts or rows > 50,000.
   - CatBoost: Approve if high-cardinality categorical columns (cardinality > 50) are present.

Return ONLY a valid JSON object matching the following structure:
{
  "approved_models": ["model1", "model2", ...],
  "rejected_models": ["model3", "model4", ...],
  "reasoning": {
    "LogisticRegression": "Explanation",
    "KNN": "Explanation",
    "SVM": "Explanation",
    "DecisionTree": "Explanation",
    "RandomForest": "Explanation",
    "ExtraTrees": "Explanation",
    "GradientBoosting": "Explanation",
    "XGBoost": "Explanation",
    "LightGBM": "Explanation",
    "CatBoost": "Explanation"
  }
}
Do not write any markdown code blocks, conversational introductions, or notes.
"""

OUTPUT_JSON_SCHEMA = {
    "name": "model_triage_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "approved_models": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "LogisticRegression", "KNN", "SVM", "DecisionTree",
                        "RandomForest", "ExtraTrees", "GradientBoosting",
                        "XGBoost", "LightGBM", "CatBoost"
                    ]
                }
            },
            "rejected_models": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "LogisticRegression", "KNN", "SVM", "DecisionTree",
                        "RandomForest", "ExtraTrees", "GradientBoosting",
                        "XGBoost", "LightGBM", "CatBoost"
                    ]
                }
            },
            "reasoning": {
                "type": "object",
                "properties": {
                    "LogisticRegression": {"type": "string"},
                    "KNN": {"type": "string"},
                    "SVM": {"type": "string"},
                    "DecisionTree": {"type": "string"},
                    "RandomForest": {"type": "string"},
                    "ExtraTrees": {"type": "string"},
                    "GradientBoosting": {"type": "string"},
                    "XGBoost": {"type": "string"},
                    "LightGBM": {"type": "string"},
                    "CatBoost": {"type": "string"}
                },
                "required": [
                    "LogisticRegression", "KNN", "SVM", "DecisionTree",
                    "RandomForest", "ExtraTrees", "GradientBoosting",
                    "XGBoost", "LightGBM", "CatBoost"
                ],
                "additionalProperties": False
            }
        },
        "required": ["approved_models", "rejected_models", "reasoning"],
        "additionalProperties": False
    }
}


def get_llm_model_triage(df: pd.DataFrame, target_col: str, task_type: str, api_key: Optional[str]) -> Dict[str, Any]:
    """
    Extracts meta-features, queries the Groq API to triage models, and returns routing results.
    Falls back to a standard default routing on API/key failures.
    """
    # 1. Fallback Defaults
    all_candidates = [
        "LogisticRegression", "KNN", "SVM", "DecisionTree",
        "RandomForest", "ExtraTrees", "GradientBoosting",
        "XGBoost", "LightGBM", "CatBoost"
    ]
    
    # Initialize basic fallback
    fallback_approved = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
    fallback_rejected = ["KNN", "SVM", "DecisionTree", "ExtraTrees"]
    
    if len(df) > 20000 or len(df.columns) > 50:
        if "KNN" in fallback_approved: fallback_approved.remove("KNN")
        if "KNN" not in fallback_rejected: fallback_rejected.append("KNN")
    if len(df) > 15000:
        if "SVM" in fallback_approved: fallback_approved.remove("SVM")
        if "SVM" not in fallback_rejected: fallback_rejected.append("SVM")
        
    default_res = {
        "approved_models": fallback_approved,
        "rejected_models": [m for m in all_candidates if m not in fallback_approved],
        "reasoning": {
            m: "Approved as default baseline model suitability." if m in fallback_approved else "Rejected by default heuristic due to size/computational bounds."
            for m in all_candidates
        }
    }
    
    if not api_key:
        logger.warning("No API key provided for LLM Triage. Returning heuristic defaults.")
        return default_res
        
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed. Returning heuristic defaults.")
        return default_res

    # 2. Extract Metadata Payload
    num_rows = len(df)
    num_cols = len(df.columns)
    num_numeric_cols = len(df.select_dtypes(include=["number"]).columns)
    num_categorical_cols = len(df.select_dtypes(exclude=["number"]).columns)
    missing_pct = round(df.isnull().mean().mean() * 100, 2)
    
    # High cardinality detection (> 50 unique values in object/categorical cols)
    high_card_features = []
    for c in df.select_dtypes(exclude=["number"]).columns:
        if c != target_col:
            card = df[c].nunique()
            if card > 50:
                high_card_features.append(f"{c} (cardinality: {card})")
                
    # Text feature count (unstructured columns)
    text_features_count = 0
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c == target_col:
            continue
        sample_vals = df[c].dropna().head(100)
        if not sample_vals.empty:
            avg_len = sample_vals.astype(str).str.len().mean()
            if avg_len > 50 and df[c].nunique() > 0.8 * len(sample_vals):
                text_features_count += 1
                
    # Class imbalance calculation
    minority_pct = 0.0
    if task_type in ["binary", "multiclass"] and target_col in df.columns:
        counts = df[target_col].dropna().value_counts()
        if not counts.empty:
            minority_pct = round((counts.min() / counts.sum()) * 100, 2)

    # 3. Format Prompt
    user_prompt = f"""Analyze this dataset metadata profile:
- Task Type: {task_type}
- Rows: {num_rows}
- Columns: {num_cols}
- Numeric Columns Count: {num_numeric_cols}
- Categorical Columns Count: {num_categorical_cols}
- Class Balance (Minority Class %): {minority_pct}%
- Missing Value Percentage: {missing_pct}%
- High-Cardinality Categorical Features (cardinality > 50): {high_card_features}
- Text / Unstructured Features Count: {text_features_count}
"""

    try:
        client = Groq(api_key=api_key)
        t0 = time.time()
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": OUTPUT_JSON_SCHEMA
            },
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )
        
        latency = int((time.time() - t0) * 1000)
        logger.info("Groq Model Triage completed in %dms", latency)
        
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        
        # Validation checks
        if "approved_models" not in result or not result["approved_models"]:
            logger.warning("LLM returned empty approved_models. Using defaults.")
            return default_res
            
        # Ensure all candidate keys exist in reasoning dict
        for m in all_candidates:
            if m not in result.get("reasoning", {}):
                if "reasoning" not in result:
                    result["reasoning"] = {}
                result["reasoning"][m] = "Approved." if m in result["approved_models"] else "Rejected."
                
        return result
        
    except Exception as e:
        logger.error("LLM model triage API call failed: %s. Returning defaults.", e)
        return default_res
