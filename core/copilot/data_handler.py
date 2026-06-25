"""
core/copilot/data_handler.py
============================
Data loading and schema extraction for Voice Data Copilot.
"""

import pandas as pd
import numpy as np
import json

MAX_ROWS_FOR_CONTEXT = 100_000   # Sample cap for very large datasets
ROW_LIMIT_WARNING    = 500_000   # Warn the user above this threshold
SAMPLE_ROWS          = 500       # How many rows to include in schema preview


def extract_schema(df: pd.DataFrame) -> dict:
    """
    Extract a comprehensive schema dictionary from a DataFrame.
    """
    rows, cols = df.shape

    # Per-column metadata
    columns_meta = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct   = round(null_count / rows * 100, 1) if rows > 0 else 0.0
        columns_meta.append({
            "name":       col,
            "dtype":      str(df[col].dtype),
            "null_count": null_count,
            "null_pct":   null_pct,
        })

    # Numeric summary (describe)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        desc = numeric_df.describe().round(4)
        desc_clean = desc.replace({np.nan: None})
        numeric_summary = json.loads(desc_clean.to_json())
    else:
        numeric_summary = {}

    # Sample rows — replace NaN with None for clean serialization via roundtrip
    sample = df.head(SAMPLE_ROWS)
    sample_clean = sample.replace({np.nan: None})
    sample_rows = json.loads(sample_clean.to_json(orient="records", date_format="iso"))

    # Duplicate row count
    duplicate_rows = int(df.duplicated().sum())

    return {
        "shape":           (rows, cols),
        "columns":         columns_meta,
        "numeric_summary": numeric_summary,
        "sample_rows":     sample_rows,
        "duplicate_rows":  duplicate_rows,
    }


def schema_to_prompt_text(schema: dict) -> str:
    """
    Convert a schema dict into a clean, highly compressed plain-text block
    suitable for injection into an LLM system prompt.
    """
    rows, cols = schema["shape"]
    lines = [
        f"Dataset shape: {rows:,} rows and {cols} columns.",
        f"Duplicate rows: {schema['duplicate_rows']:,}.",
        "",
        "Columns (name | dtype | missing values):",
    ]

    for col in schema["columns"]:
        null_info = (
            f"{col['null_count']:,} missing ({col['null_pct']}%)"
            if col["null_count"] > 0
            else "no missing values"
        )
        lines.append(f"  - {col['name']} [{col['dtype']}] — {null_info}")

    # Eager numeric statistics are omitted to conserve tokens.
    # The agent can run df.describe() dynamically if it needs statistical summaries.

    # Sample rows (Limit to max 2 rows, compressed in CSV format)
    if schema["sample_rows"]:
        lines.append("")
        lines.append("First 2 sample rows (CSV format):")
        cols_list = [col["name"] for col in schema["columns"]]
        lines.append(",".join(cols_list))
        for row in schema["sample_rows"][:2]:
            row_vals = []
            for col_name in cols_list:
                val = row.get(col_name, "")
                if val is None:
                    row_vals.append("")
                else:
                    val_str = str(val).replace("\n", " ").replace("\r", " ")
                    if "," in val_str or '"' in val_str:
                        val_str = f'"{val_str.replace(chr(34), chr(34)+chr(34))}"'
                    row_vals.append(val_str)
            lines.append(",".join(row_vals))

    return "\n".join(lines)
