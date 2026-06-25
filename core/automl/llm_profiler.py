"""
core/automl/llm_profiler.py — SOL Agentic Data Profiler
Groq-powered LLM layer for semantic dataset analysis.
Provider: Groq | Model: openai/gpt-oss-120b

Streamlit dependencies removed — runs as plain Python in FastAPI context.
"""

import json
import hashlib
import logging
import os
import time
from typing import Optional, Dict, Any, List

import pandas as pd

logger = logging.getLogger("SOL.AgenticProfiler")

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MAX_TOKENS = 2048
GROQ_TEMPERATURE = 0.1


def resolve_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the Groq API key with priority order:
      1. Caller-provided key (from HTTP request)
      2. Environment variable GROQ_API_KEY
    Returns None if no key is found.
    """
    if provided_key and provided_key.strip():
        return provided_key.strip()
    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


def extract_data_signature(df: pd.DataFrame, max_sample_rows: int = 5) -> Dict[str, Any]:
    """
    Extracts a minimal, token-efficient dataset fingerprint for LLM analysis.
    """
    signature = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": []
    }
    for col in df.columns:
        samples = df[col].dropna().head(max_sample_rows).tolist()
        safe_samples = []
        for v in samples:
            if isinstance(v, (int, float, str, bool)):
                safe_samples.append(v)
            else:
                safe_samples.append(str(v))
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "null_pct": round(df[col].isnull().mean() * 100, 1),
            "sample_values": safe_samples
        }
        signature["columns"].append(col_info)
    return signature


def compute_dataset_hash(df: pd.DataFrame) -> str:
    hash_input = f"{df.shape}_{list(df.columns)}_{df.dtypes.tolist()}"
    return hashlib.md5(hash_input.encode()).hexdigest()


SYSTEM_PROMPT = """You are "SOL", an expert AI Data Profiler for the "Al-Dalil" Automated Machine Learning (AutoML) platform.
You analyze dataset metadata to provide intelligent preprocessing recommendations.

Your task is to analyze a "Data Signature" — a summary of a dataset's column names, data types,
unique value counts, null percentages, and sample values — and return structured recommendations.

RULES:
1. IDENTIFY the most likely TARGET column for ML prediction based on:
   - Column name semantics (e.g., "disposition", "survived", "price", "label", "class", "churn", "fraud")
   - Data characteristics (low-cardinality categorical = classification, continuous numeric = regression)
   - Domain context (astronomy, finance, healthcare, etc.)

2. DETECT DATA LEAKAGE by identifying columns that:
   - Are derived from or strongly correlated with the target (e.g., preliminary human judgments,
     confidence scores for the target, post-event indicators)
   - Are unique identifiers (IDs, row numbers, names) that do not generalize
   - Have near-zero predictive value (constant columns, administrative timestamps)

3. INFER the correct ML TASK TYPE:
   - "binary" if the target has exactly 2 classes
   - "multiclass" if the target has 3+ discrete classes
   - "regression" if the target is continuous numeric

4. PROVIDE REASONING in both Arabic and English:
   - "reasoning" fields must be bilingual: Arabic first, then English in parentheses.
   - "summary" field must also be bilingual.

IMPORTANT: Be conservative with blacklisting. Only flag columns you are confident represent
leakage, IDs, or noise. When in doubt, keep the column and explain the uncertainty."""

OUTPUT_JSON_SCHEMA = {
    "name": "agentic_profiler_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "suggested_target": {
                "type": "object",
                "properties": {
                    "column_name": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reasoning": {"type": "string"}
                },
                "required": ["column_name", "confidence", "reasoning"],
                "additionalProperties": False
            },
            "task_type": {"type": "string", "enum": ["binary", "multiclass", "regression"]},
            "blacklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string"},
                        "reason_category": {
                            "type": "string",
                            "enum": ["data_leakage", "unique_identifier", "post_event", "high_cardinality_noise", "constant"]
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["column_name", "reason_category", "reasoning"],
                    "additionalProperties": False
                }
            },
            "domain_detected": {"type": "string"},
            "summary": {"type": "string"},
            "force_numeric": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["suggested_target", "task_type", "blacklist", "domain_detected", "summary", "force_numeric"],
        "additionalProperties": False
    }
}


def analyze(df: pd.DataFrame, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Sends a Data Signature to Groq and returns structured profiling results.
    No Streamlit dependency — works in any Python context.
    """
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq>=0.13.0")
        return None

    signature = extract_data_signature(df)
    signature_json = json.dumps(signature, ensure_ascii=False, default=str)
    user_prompt = f"""Analyze the following Data Signature and return your recommendations as structured JSON.

DATA SIGNATURE:
{signature_json}"""

    try:
        client = Groq(api_key=api_key)
        t_start = time.time()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_schema", "json_schema": OUTPUT_JSON_SCHEMA},
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )
        latency_ms = int((time.time() - t_start) * 1000)
        logger.info("Groq API response received in %dms", latency_ms)
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        result["_meta"] = {
            "model": GROQ_MODEL,
            "latency_ms": latency_ms,
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0)
        }
        if result["suggested_target"]["column_name"] not in df.columns:
            logger.warning("LLM suggested target '%s' not found in DataFrame. Invalidating.", result["suggested_target"]["column_name"])
            return None
        result["blacklist"] = [entry for entry in result["blacklist"] if entry["column_name"] in df.columns]
        return result
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Groq response as JSON: %s", e)
        return None
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return None
