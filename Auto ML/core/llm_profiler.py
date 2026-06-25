"""
core/llm_profiler.py — SOL Agentic Data Profiler
Groq-powered LLM layer for semantic dataset analysis.
Provider: Groq | Model: openai/gpt-oss-120b
"""

import json
import hashlib
import logging
import os
import time
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st

logger = logging.getLogger("SOL.AgenticProfiler")

# Model Configuration
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MAX_TOKENS = 2048
GROQ_TEMPERATURE = 0.1  # Low temperature for deterministic structured output


# ─────────────────────────────────────────────
# 1. API Key Resolution (UI > secrets.toml > env)
# ─────────────────────────────────────────────

def resolve_api_key() -> Optional[str]:
    """
    Resolves the Groq API key with strict priority order:
      1. User-provided key via Streamlit sidebar input (session_state)
      2. .streamlit/secrets.toml → [groq] section
      3. Environment variable GROQ_API_KEY
    Returns None if no key is found.
    """
    # Priority 1: UI-provided key (highest priority per user request A3)
    ui_key = st.session_state.get("groq_api_key_input", "")
    if ui_key and ui_key.strip():
        return ui_key.strip()

    # Priority 2: Streamlit secrets.toml
    try:
        key = st.secrets["groq"]["GROQ_API_KEY"]
        if key and key.strip():
            return key.strip()
    except (KeyError, FileNotFoundError, AttributeError):
        pass

    # Priority 3: Environment variable
    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip()

    return None


# ─────────────────────────────────────────────
# 2. Data Signature Extraction
# ─────────────────────────────────────────────

def extract_data_signature(df: pd.DataFrame, max_sample_rows: int = 5) -> Dict[str, Any]:
    """
    Extracts a minimal, token-efficient dataset fingerprint for LLM analysis.
    Compresses any dataset to ~300-800 tokens regardless of original size.
    """
    signature = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": []
    }
    for col in df.columns:
        # Convert sample values to JSON-safe primitives
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
    """Computes a stable hash for caching LLM results per unique dataset."""
    hash_input = f"{df.shape}_{list(df.columns)}_{df.dtypes.tolist()}"
    return hashlib.md5(hash_input.encode()).hexdigest()


# ─────────────────────────────────────────────
# 3. Prompt Engineering
# ─────────────────────────────────────────────

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
   - Example: "هذا العمود يمثل التصنيف النهائي للكواكب (This column represents the final planet classification)"
   - "summary" field must also be bilingual.

IMPORTANT: Be conservative with blacklisting. Only flag columns you are confident represent
leakage, IDs, or noise. When in doubt, keep the column and explain the uncertainty."""


# ─────────────────────────────────────────────
# 4. JSON Schema for Strict Output
# ─────────────────────────────────────────────

OUTPUT_JSON_SCHEMA = {
    "name": "agentic_profiler_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "suggested_target": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "The exact column name from the dataset."
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Bilingual Arabic/English explanation."
                    }
                },
                "required": ["column_name", "confidence", "reasoning"],
                "additionalProperties": False
            },
            "task_type": {
                "type": "string",
                "enum": ["binary", "multiclass", "regression"]
            },
            "blacklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {
                            "type": "string"
                        },
                        "reason_category": {
                            "type": "string",
                            "enum": [
                                "data_leakage",
                                "unique_identifier",
                                "post_event",
                                "high_cardinality_noise",
                                "constant"
                            ]
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Bilingual Arabic/English explanation."
                        }
                    },
                    "required": ["column_name", "reason_category", "reasoning"],
                    "additionalProperties": False
                }
            },
            "domain_detected": {
                "type": "string"
            },
            "summary": {
                "type": "string",
                "description": "Bilingual 2-3 sentence dataset assessment."
            }
        },
        "required": [
            "suggested_target",
            "task_type",
            "blacklist",
            "domain_detected",
            "summary"
        ],
        "additionalProperties": False
    }
}


# ─────────────────────────────────────────────
# 5. Core Analysis Function
# ─────────────────────────────────────────────

def analyze(df: pd.DataFrame, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Sends a Data Signature to Groq and returns structured profiling results.
    
    Args:
        df: The uploaded DataFrame.
        api_key: Validated Groq API key.
        
    Returns:
        Parsed JSON dict with profiling results, or None on failure.
    """
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq>=0.13.0")
        return None

    # Extract lightweight data fingerprint
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
            response_format={
                "type": "json_schema",
                "json_schema": OUTPUT_JSON_SCHEMA
            },
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )

        latency_ms = int((time.time() - t_start) * 1000)
        logger.info("Groq API response received in %dms", latency_ms)

        # Parse the structured JSON output
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)

        # Attach metadata
        result["_meta"] = {
            "model": GROQ_MODEL,
            "latency_ms": latency_ms,
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0)
        }

        # Validate that suggested target actually exists in the DataFrame
        if result["suggested_target"]["column_name"] not in df.columns:
            logger.warning(
                "LLM suggested target '%s' not found in DataFrame columns. Invalidating.",
                result["suggested_target"]["column_name"]
            )
            return None

        # Validate blacklist columns exist
        result["blacklist"] = [
            entry for entry in result["blacklist"]
            if entry["column_name"] in df.columns
        ]

        return result

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Groq response as JSON: %s", e)
        return None
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return None


# ─────────────────────────────────────────────
# 6. Cached Wrapper (keyed by dataset hash)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def analyze_cached(dataset_hash: str, columns_json: str, dtypes_json: str,
                   shape_tuple: tuple, head_json: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Cached wrapper around analyze(). Keyed by dataset hash to prevent
    redundant Groq API calls on Streamlit reruns.
    
    We reconstruct the DataFrame from serialized components because
    st.cache_data requires hashable arguments.
    """
    import io
    head_df = pd.read_json(io.StringIO(head_json))
    
    # For the LLM, we only need the signature — not the full data.
    # But we need column metadata from the full dataset.
    # We reconstruct a minimal representative DataFrame.
    columns = json.loads(columns_json)
    
    # Build a synthetic DataFrame with correct schema for signature extraction
    # using the head data which has real samples
    result_df = head_df
    
    return analyze(result_df, api_key)


def run_profiler(df: pd.DataFrame, api_key: str) -> Optional[Dict[str, Any]]:
    """
    High-level entry point that handles caching transparently.
    Call this from app.py.
    """
    dataset_hash = compute_dataset_hash(df)
    
    # Serialize components for st.cache_data (must be hashable)
    # Use enough rows to capture the data signature accurately
    sample_size = min(len(df), 200)
    head_df = df.head(sample_size)
    
    return analyze_cached(
        dataset_hash=dataset_hash,
        columns_json=json.dumps(list(df.columns)),
        dtypes_json=json.dumps({c: str(d) for c, d in df.dtypes.items()}),
        shape_tuple=df.shape,
        head_json=head_df.to_json(),
        api_key=api_key
    )


# ─────────────────────────────────────────────
# 7. Reason Category Display Helpers
# ─────────────────────────────────────────────

CATEGORY_LABELS = {
    "data_leakage": ("🔴 تسرب بيانات", "🔴 Data Leakage"),
    "unique_identifier": ("🟡 معرف فريد", "🟡 Unique Identifier"),
    "post_event": ("🟠 متغير ما بعد الحدث", "🟠 Post-Event Variable"),
    "high_cardinality_noise": ("🔵 ضوضاء عالية التفرد", "🔵 High-Cardinality Noise"),
    "constant": ("⚪ ثابت / بدون تباين", "⚪ Constant / Zero Variance"),
}

CONFIDENCE_ICONS = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🔴",
}


def get_category_label(category: str, is_arabic: bool = False) -> str:
    """Returns a display-friendly label for a blacklist reason category."""
    labels = CATEGORY_LABELS.get(category, ("⬜ غير معروف", "⬜ Unknown"))
    return labels[0] if is_arabic else labels[1]


def get_confidence_icon(confidence: str) -> str:
    """Returns an emoji icon for confidence level."""
    return CONFIDENCE_ICONS.get(confidence, "⚪")
