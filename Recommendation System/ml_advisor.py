"""
MLAdvisor Module
================
A self-contained module that analyses a cleaned DataFrame and returns
top-3 ML model recommendations powered by Llama 3 via the Groq API.

Usage
-----
    from ml_advisor import get_recommendations
    import pandas as pd

    df = pd.read_csv("titanic_clean.csv")
    result = get_recommendations(df, target_column="Survived")
    print(result)   # returns a dict (JSON-compatible)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  ⚙️  CONFIG  – edit the values below directly
# ─────────────────────────────────────────────────────────────────────────────
CONFIG: dict[str, Any] = {
    # ── Groq settings ──────────────────────────────────────────────────────
    "GROQ_API_KEY":  "your_api_key_here",
    "GROQ_MODEL":    "llama-3.3-70b-versatile",   # or "llama3-8b-8192"
    "GROQ_API_URL":  "https://api.groq.com/openai/v1/chat/completions",

    # ── Shared LLM settings ─────────────────────────────────────────────────
    "LLM_TEMPERATURE": 0.2,
    "LLM_MAX_TOKENS":  1024,
}

# ─────────────────────────────────────────────────────────────────────────────
#  📊  PART 1 – The Metadata Profiler  ("The Eye")
# ─────────────────────────────────────────────────────────────────────────────

def _profile_metadata(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Extract a rich metadata packet from *df* relative to *target_column*.

    Returns
    -------
    dict with four top-level keys:
        scale_info, feature_profiling, target_anatomy, health_metrics
    """
    # ── Guard: target column must exist ────────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_column]
    target_series = df[target_column]

    # ── 1. Scale Info ───────────────────────────────────────────────────────
    scale_info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_feature_cols": len(feature_cols),
    }

    # ── 2. Feature Profiling ────────────────────────────────────────────────
    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    binary_cols: list[str] = []

    for col in feature_cols:
        n_unique = df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
        if n_unique == 2:
            binary_cols.append(col)

    feature_profiling = {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "binary_cols": binary_cols,
        "n_numerical": len(numerical_cols),
        "n_categorical": len(categorical_cols),
        "n_binary": len(binary_cols),
        "cat_to_num_ratio": round(
            len(categorical_cols) / max(len(numerical_cols), 1), 4
        ),
    }

    # ── 3. Target Anatomy ───────────────────────────────────────────────────
    n_unique_target = target_series.nunique(dropna=True)
    is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

    # Heuristic: if numeric with many unique values → regression
    if is_numeric_target and n_unique_target > 20:
        task_type = "Regression"
        class_balance = None
    elif n_unique_target == 2:
        task_type = "Binary Classification"
        vc = target_series.value_counts(normalize=True).round(4).to_dict()
        class_balance = {str(k): float(v) for k, v in vc.items()}
    else:
        task_type = "Multi-class Classification"
        vc = target_series.value_counts(normalize=True).round(4).to_dict()
        class_balance = {str(k): float(v) for k, v in vc.items()}

    target_anatomy = {
        "task_type": task_type,
        "target_dtype": str(target_series.dtype),
        "n_unique_values": int(n_unique_target),
        "class_balance": class_balance,
    }

    # ── 4. Health Metrics ───────────────────────────────────────────────────
    missing_rates = df.isnull().mean().round(4).to_dict()
    missing_rates = {str(k): float(v) for k, v in missing_rates.items()}
    overall_missing_pct = round(df.isnull().values.mean() * 100, 4)

    skewness: dict[str, float] = {}
    for col in numerical_cols[:20]:          # cap at 20 to keep prompt lean
        col_data = df[col].dropna()
        if len(col_data) > 2:
            try:
                skew_val = float(stats.skew(col_data))
                skewness[col] = round(skew_val, 4)
            except Exception:
                skewness[col] = None  # type: ignore[assignment]

    health_metrics = {
        "overall_missing_pct": overall_missing_pct,
        "per_column_missing_rate": missing_rates,
        "numerical_skewness": skewness,
    }

    return {
        "scale_info": scale_info,
        "feature_profiling": feature_profiling,
        "target_anatomy": target_anatomy,
        "health_metrics": health_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  🧠  PART 2 – The Reasoning Engine  ("The Mind")
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert Machine Learning Solutions Architect. Your task is to provide \
a high-level strategic recommendation for building a predictive model based on \
the provided dataset metadata.

Analyze the following constraints:
1. Data Volume: Assess if the dataset size requires high-bias/low-variance models \
or can support high-complexity ensembles.
2. Feature Mixture: Evaluate the ratio of categorical vs. numerical features to \
suggest models with native handling of specific types.
3. Target Nature: Determine the exact ML task (Binary/Multi-class Classification, \
Regression, or Clustering).

Your Output Requirements:
- Provide EXACTLY 3 model recommendations.
- Respond ONLY with a strict JSON object — no markdown, no code fences, no prose.

Output Format (strict JSON):
{
  "task_identified": "...",
  "recommendations": [
    {
      "model": "...",
      "score": 0,
      "pros": "...",
      "cons": "..."
    }
  ]
}
"""


def _build_user_message(metadata_packet: dict[str, Any]) -> str:
    """Serialise the metadata packet into a human-readable user message."""
    packet_json = json.dumps(metadata_packet, indent=2, ensure_ascii=False)
    return (
        "Please analyse the following dataset metadata packet and provide your "
        "top-3 ML model recommendations in the required JSON format.\n\n"
        f"```json\n{packet_json}\n```"
    )


def _call_groq(user_message: str) -> str:
    """Call the Groq cloud API and return the raw LLM response string."""
    payload = json.dumps(
        {
            "model": CONFIG["GROQ_MODEL"],
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": CONFIG["LLM_TEMPERATURE"],
            "max_tokens": CONFIG["LLM_MAX_TOKENS"],
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        CONFIG["GROQ_API_URL"],
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['GROQ_API_KEY']}",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise ValueError(f"Groq API Error {e.code}: {error_body}") from e

    return body["choices"][0]["message"]["content"]


def _call_llm(user_message: str) -> dict[str, Any]:
    """
    Call the Groq backend, parse JSON, and return the recommendation dict.
    """
    raw = _call_groq(user_message)

    # Strip accidental markdown fences before parsing
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON output. Raw response:\n{raw}"
        ) from exc

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
#  🔗  PART 3 – Integration & Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Main public entry-point for the MLAdvisor module.

    Parameters
    ----------
    df : pd.DataFrame
        A **cleaned** DataFrame (no raw, unprocessed data).
    target_column : str
        The name of the prediction target column inside *df*.

    Returns
    -------
    dict
        A JSON-compatible dictionary::

            {
                "status": "success",
                "metadata": { ... },
                "task_identified": "...",
                "recommendations": [
                    {"model": "...", "score": 85, "pros": "...", "cons": "..."},
                    ...   # 3 items total
                ]
            }

        On error::

            {"status": "error", "error_type": "...", "message": "..."}
    """
    # ── Validation ──────────────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        return _error_response("InvalidInputError", "The 'df' argument must be a pandas DataFrame.")

    if df.empty:
        return _error_response("EmptyDataError", "The provided DataFrame is empty – nothing to analyse.")

    if not isinstance(target_column, str) or not target_column.strip():
        return _error_response("InvalidTargetError", "target_column must be a non-empty string.")

    # ── Step 1: Metadata Profiling ──────────────────────────────────────────
    try:
        metadata = _profile_metadata(df, target_column)
    except ValueError as exc:
        return _error_response("ProfilerError", str(exc))
    except Exception as exc:
        return _error_response("UnexpectedProfilerError", f"Metadata profiling failed unexpectedly: {exc}")

    # ── Step 2: LLM Reasoning ───────────────────────────────────────────────
    try:
        user_message = _build_user_message(metadata)
        llm_result = _call_llm(user_message)
    except ValueError as exc:
        return _error_response("LLMParsingError", str(exc))
    except Exception as exc:
        return _error_response("LLMCallError", f"Failed to get a response from the LLM backend: {exc}")

    # ── Step 3: Compose Final Response ─────────────────────────────────────
    return {
        "status": "success",
        "metadata": metadata,
        "task_identified": llm_result.get("task_identified", "Unknown"),
        "recommendations": llm_result.get("recommendations", []),
    }


def _error_response(error_type: str, message: str) -> dict[str, Any]:
    """Return a standardised error envelope."""
    return {"status": "error", "error_type": error_type, "message": message}


# ─────────────────────────────────────────────────────────────────────────────
#  🧪  Quick smoke-test  (run:  python ml_advisor.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib
    import sys


    base = pathlib.Path(__file__).parent
    csv_path = next(
        (p for p in (base / "test 1.csv", base / "data.csv") if p.exists()),
        None,
    )

    if csv_path is None:
        print("[Error] CSV file not found. Execution stopped.")
        sys.exit(1)

    demo_df = pd.read_csv(csv_path)
    candidates = [c for c in demo_df.columns if c.lower() in ("survived", "target", "label", "class")]
    target = candidates[0] if candidates else demo_df.columns[-1]
    print(f"[MLAdvisor] Loaded {csv_path.name}  |  target='{target}'")

    print("[MLAdvisor] Profiling metadata …")
    meta = _profile_metadata(demo_df, target)
    print(json.dumps(meta, indent=2))

    print("\n[MLAdvisor] Calling Groq …")
    result = get_recommendations(demo_df, target)
    print(json.dumps(result, indent=2, ensure_ascii=False))
