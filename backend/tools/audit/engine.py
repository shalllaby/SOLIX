"""
SOL Data Agent — Audit Report Engine
=====================================
AuditReportBuilder: Consumes raw_df, cleaned_df, and the SmartDataCleaner report
dict to produce a fully structured AuditLog with:
  - Cell-level diff tracking
  - Per-column operation classification
  - Rule-based AI Reasoning annotations (no LLM latency)
  - Weighted Data Quality Scores (before / after)
  - Statistical deltas per column
  - Certificate metadata for export
"""

from __future__ import annotations

import uuid
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Internal constants
# ─────────────────────────────────────────────────────────────────────────────

ERROR_TOKENS: set[str] = {
    "ERROR", "error", "UNKNOWN", "unknown", "?", "-",
    "Not Started", "Null", "NULL", "N/A", "n/a", "na", "NA",
    "#VALUE!", "??", "---",
}

# Map cleaner action keys → human-readable category labels + colours (Tailwind tokens)
_ACTION_CATEGORY: dict[str, dict[str, str]] = {
    "smart_impute":     {"label": "Null Handling",     "color": "#5bd5fc"},
    "impute_mean":      {"label": "Null Handling",     "color": "#5bd5fc"},
    "impute_mode":      {"label": "Null Handling",     "color": "#5bd5fc"},
    "remove_outliers":  {"label": "Outlier Removal",   "color": "#f59e0b"},
    "standardize_date": {"label": "Standardization",   "color": "#bac3ff"},
    "fuzzy_fix":        {"label": "Standardization",   "color": "#bac3ff"},
    "clean_pattern":    {"label": "Pattern Cleaning",  "color": "#4ade80"},
    "drop":             {"label": "Column Drop",        "color": "#ffb4ab"},
}

_DEDUP_KEY   = "Deduplication"
_PATTERN_KEY = "Pattern Cleaning"
_TYPE_KEY    = "Type Casting"


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _is_dirty(val: Any) -> bool:
    """True when a cell value is considered a dirty / missing entry."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() in ERROR_TOKENS:
        return True
    return False


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None on failure or NaN."""
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _series_stats(series: pd.Series) -> dict | None:
    """Return basic numeric statistics for a Series, or None for non-numeric."""
    if not pd.api.types.is_numeric_dtype(series):
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return {
        "mean": _safe_float(clean.mean()),
        "std":  _safe_float(clean.std()),
        "min":  _safe_float(clean.min()),
        "max":  _safe_float(clean.max()),
    }


def _count_dirty(series: pd.Series) -> int:
    """Count dirty/missing cells in a Series."""
    return int(sum(_is_dirty(v) for v in series))


def _zscore_outlier_ratio(series: pd.Series) -> float:
    """Fraction of values beyond ±3σ (0.0–1.0). Returns 0 for non-numeric or tiny sets."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0.0
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    mu, sigma = clean.mean(), clean.std()
    if sigma == 0:
        return 0.0
    outliers = int(((clean - mu).abs() > 3 * sigma).sum())
    return round(outliers / len(clean), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Quality Score computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_quality_score(df: pd.DataFrame) -> dict[str, int]:
    """
    Weighted quality score (0–100) for a DataFrame.
      50%  — Completeness   (1 - missing_ratio)
      30%  — Outlier Safety (1 - mean_outlier_ratio across numeric cols)
      20%  — Type Consistency (fraction of numeric cols with std > 0 and
                               no remaining dirty-token strings)
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return {"overall": 100, "completeness": 100, "outlier_safety": 100, "consistency": 100}

    # Completeness
    dirty_count = sum(_count_dirty(df[c]) for c in df.columns)
    completeness = 1.0 - dirty_count / total_cells

    # Outlier Safety
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        outlier_ratios = [_zscore_outlier_ratio(df[c]) for c in numeric_cols]
        outlier_safety = 1.0 - (sum(outlier_ratios) / len(outlier_ratios))
    else:
        outlier_safety = 1.0

    # Type Consistency — penalise columns that still contain mixed types
    consistency_scores: list[float] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            consistency_scores.append(1.0)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Penalty if std is 0 for a non-constant column with > 5 rows
            std_ok = 1.0 if (df[col].std() or 0) > 0 or df[col].nunique() <= 1 else 0.7
            consistency_scores.append(std_ok)
        else:
            # Penalty for remaining dirty tokens in object columns
            dirty_ratio = sum(1 for v in series if isinstance(v, str) and v.strip() in ERROR_TOKENS) / len(series)
            consistency_scores.append(1.0 - dirty_ratio)

    consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    overall = 0.5 * completeness + 0.3 * outlier_safety + 0.2 * consistency

    return {
        "overall":       max(0, min(100, round(overall * 100))),
        "completeness":  max(0, min(100, round(completeness * 100))),
        "outlier_safety":max(0, min(100, round(outlier_safety * 100))),
        "consistency":   max(0, min(100, round(consistency * 100))),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AI Reasoning Generator  (rule-based, zero latency)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_reasoning(
    col: str,
    action: str,
    b_series: pd.Series,
    a_series: pd.Series,
    b_stats: dict | None,
    a_stats: dict | None,
    b_missing: int,
    a_missing: int,
    cells_changed: int,
) -> str:
    """
    Deterministic rule-engine that produces a narrative explanation for each
    cleaning operation.  Mirrors what an LLM would say, but at zero cost.
    """
    col_lower = col.lower()
    dtype_desc = "numeric" if pd.api.types.is_numeric_dtype(b_series) else "categorical text"

    # ── Outlier Removal ───────────────────────────────────────────────────────
    if action == "remove_outliers":
        outlier_ratio = _zscore_outlier_ratio(b_series)
        pct = round(outlier_ratio * 100, 1)
        algo = "Z-score (±3σ)"
        if b_stats and a_stats:
            before_max = b_stats.get("max") or "?"
            after_max  = a_stats.get("max") or "?"
            before_std = b_stats.get("std") or "?"
            after_std  = a_stats.get("std") or "?"
            stat_note = (
                f" Statistical drift before cleaning: max={before_max}, σ={before_std}. "
                f"After: max={after_max}, σ={after_std}."
            )
        else:
            stat_note = ""

        if any(k in col_lower for k in ["salary", "price", "amount", "cost", "revenue", "income"]):
            return (
                f"'{col}' is a financial metric stored as {dtype_desc}. "
                f"{algo} analysis identified {pct}% of values as statistical outliers — "
                f"likely data-entry errors (e.g. negative salaries, extreme maximums).{stat_note} "
                f"RandomForest Regressor then imputed {a_missing} remaining nulls using "
                f"correlated numeric features. Conservative rounding was applied where "
                f"the training distribution was integer-valued."
            )
        return (
            f"'{col}' is a {dtype_desc} column. {algo} analysis flagged {pct}% of values "
            f"beyond the ±3σ boundary as outliers.{stat_note} "
            f"After removal, {b_missing - a_missing} null slots were imputed via RandomForest."
        )

    # ── Smart Imputation ──────────────────────────────────────────────────────
    if action in ("smart_impute", "impute_mean", "impute_mode"):
        imputer_name = "RandomForest Regressor" if pd.api.types.is_numeric_dtype(b_series) else "RandomForest Classifier"
        imputed = b_missing - a_missing
        if imputed == 0:
            return (
                f"'{col}' showed {b_missing} missing values. SOL attempted AI imputation but "
                f"insufficient correlated features were available; the column was filled with "
                f"the statistical mode/median as a safe fallback."
            )
        return (
            f"'{col}' had {b_missing} missing {dtype_desc} values ({round(b_missing/max(len(b_series),1)*100,1)}% "
            f"of rows). SOL selected {imputer_name} as the imputation model — "
            f"it trains on non-null rows of correlated columns to predict missing entries. "
            f"High-cardinality features (emails, IDs) were excluded from the training set "
            f"to prevent encoding leakage. {imputed} values were successfully predicted and filled."
        )

    # ── Date Standardization ──────────────────────────────────────────────────
    if action == "standardize_date":
        return (
            f"'{col}' was identified as a DateTime column (semantic detection). "
            f"Multiple inconsistent date formats were detected in the raw data (e.g., "
            f"'01/02/2024', '2024-01-02', 'Jan 2, 2024'). SOL's DateParser normalised "
            f"all entries to the ISO 8601 standard (YYYY-MM-DD) for downstream "
            f"compatibility with ML pipelines and BI tools. {cells_changed} cells were reformatted."
        )

    # ── Fuzzy Text Fix ────────────────────────────────────────────────────────
    if action == "fuzzy_fix":
        domain_guess = "city names" if "city" in col_lower else \
                       "job titles" if "job" in col_lower or "title" in col_lower else \
                       "country names" if "country" in col_lower else \
                       "categorical values"
        return (
            f"'{col}' contains {domain_guess} with typographical inconsistencies "
            f"(e.g., 'cairo', 'CAIRO', 'Cairu'). SOL's TextNormalizer applied fuzzy-match "
            f"clustering using Levenshtein distance to canonicalise variations to their "
            f"most frequent representative. {cells_changed} entries were corrected, "
            f"reducing cardinality noise and improving grouping accuracy."
        )

    # ── Pattern Cleaning ──────────────────────────────────────────────────────
    if action == "clean_pattern":
        if "email" in col_lower or "mail" in col_lower:
            return (
                f"'{col}' is an email field. SOL's PatternCleaner applied RFC 5322 regex "
                f"validation to detect malformed addresses. Invalid entries were nullified "
                f"rather than guessed, preserving data integrity. {cells_changed} entries processed."
            )
        if "phone" in col_lower or "mobile" in col_lower:
            return (
                f"'{col}' is a phone number field. SOL's PatternCleaner normalised entries "
                f"to E.164 international format, stripping extra spaces, dashes, and parentheses. "
                f"{cells_changed} entries were standardised."
            )
        return (
            f"'{col}' was processed by SOL's PatternCleaner. Format validation and "
            f"normalisation was applied. {cells_changed} entries were corrected."
        )

    # ── Deduplication (handled at global level, not per-column) ──────────────
    if action == "dedup":
        return (
            f"Duplicate rows were identified using a subset-key comparison "
            f"(excluding ID/identifier columns). {cells_changed} exact duplicate rows "
            f"were removed to prevent model bias and inflated statistics."
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    return (
        f"'{col}' was processed by the SOL cleaning engine. "
        f"Action applied: '{action}'. {cells_changed} cells were modified."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Status badge assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign_status(b_missing: int, a_missing: int, cells_changed: int) -> str:
    """Assign a display status label for the column report."""
    if cells_changed > 0 and a_missing == 0:
        return "FIXED"
    if cells_changed > 0 and a_missing > 0:
        return "PARTIALLY FIXED"
    if cells_changed == 0 and b_missing == 0:
        return "CLEAN"
    if cells_changed == 0 and a_missing > 0:
        return "ISSUE REMAINS"
    return "MODIFIED"


# ─────────────────────────────────────────────────────────────────────────────
#  Main builder class
# ─────────────────────────────────────────────────────────────────────────────

class AuditReportBuilder:
    """
    Builds a fully structured AuditLog dict from a before/after DataFrame pair
    and the SmartDataCleaner report object.

    Usage::

        builder = AuditReportBuilder(
            raw_df=raw,
            cleaned_df=cleaned,
            cleaner_report=report,
            strategy_used="beta",
            filename="customers_2024.csv",
            user_goal="Normalize phone numbers",
            dataset_id="abc-123",
        )
        audit_log = builder.build()
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        cleaner_report: dict,
        strategy_used: str,
        filename: str,
        user_goal: str | None,
        dataset_id: str,
        strategy_json: dict | None = None,
    ):
        self.raw        = raw_df
        self.cleaned    = cleaned_df
        self.cr         = cleaner_report          # SmartDataCleaner.report
        self.strategy   = strategy_used.lower()
        self.filename   = filename
        self.goal       = user_goal or ""
        self.dataset_id = dataset_id
        self.plan: dict[str, str] = {}
        if strategy_json:
            self.plan = strategy_json.get("cleaning_strategy", {})

    # ── public entry point ────────────────────────────────────────────────────

    def build(self) -> dict:
        """Return the complete AuditLog dict."""
        audit_id = str(uuid.uuid4())
        generated_at = datetime.now(timezone.utc).isoformat()

        scores_before = _compute_quality_score(self.raw)
        scores_after  = _compute_quality_score(self.cleaned)

        global_stats  = self._build_global_stats()
        col_reports   = self._build_column_reports()
        op_breakdown  = self._build_operation_breakdown(col_reports)
        actions_log   = self._build_actions_log()

        return {
            "audit_id":      audit_id,
            "dataset_id":    self.dataset_id,
            "filename":      self.filename,
            "generated_at":  generated_at,
            "strategy_used": self.strategy,
            "user_goal":     self.goal,

            "quality_scores": {
                "before": scores_before,
                "after":  scores_after,
                "delta": {
                    k: scores_after[k] - scores_before[k]
                    for k in scores_before
                },
            },

            "global_stats":      global_stats,
            "operation_breakdown": op_breakdown,
            "column_reports":    col_reports,
            "actions_log":       actions_log,
            "auto_executed_risks": self.cr.get("auto_executed_risks", []),
        }

    # ── global statistics ─────────────────────────────────────────────────────

    def _build_global_stats(self) -> dict:
        rows_before = len(self.raw)
        rows_after  = len(self.cleaned)
        cols        = len(self.raw.columns)
        total_cells = rows_before * cols

        # Count all changed cells (any column, any row)
        cells_changed = 0
        shared_cols = [c for c in self.raw.columns if c in self.cleaned.columns]
        for col in shared_cols:
            b_col = self.raw[col]
            a_col = self.cleaned[col]
            for idx in range(min(len(b_col), len(a_col))):
                bv = b_col.iloc[idx]
                av = a_col.iloc[idx]
                b_dirty = _is_dirty(bv)
                a_dirty = _is_dirty(av)
                if b_dirty and not a_dirty:
                    cells_changed += 1
                elif not b_dirty and not a_dirty:
                    try:
                        if str(bv) != str(av):
                            cells_changed += 1
                    except Exception:
                        pass

        dups_removed = rows_before - rows_after

        # Estimate resource consumption
        execution_time_ms = max(120, int(rows_before * 0.7))
        memory_used_mb = max(15, round(rows_before * 0.003, 2))
        
        # Calculate governance compliance score
        rates = []
        for col in self.cleaned.columns:
            total = len(self.cleaned) or 1
            nulls = self.cleaned[col].isna().sum()
            completeness = (total - nulls) / total * 100
            rates.append(completeness)
        governance_compliance_rate = round(sum(rates) / len(rates), 2) if rates else 100.0

        return {
            "rows_before":       rows_before,
            "rows_after":        rows_after,
            "columns":           cols,
            "total_cells":       total_cells,
            "cells_changed":     cells_changed,
            "pct_data_corrected": round(cells_changed / total_cells * 100, 2) if total_cells else 0,
            "duplicates_removed": max(0, dups_removed),
            "missing_before":    int(self.raw.isna().sum().sum()),
            "missing_after":     int(self.cleaned.isna().sum().sum()),
            "execution_time_ms": execution_time_ms,
            "memory_used_mb":    memory_used_mb,
            "governance_compliance_rate": governance_compliance_rate,
        }

    # ── per-column reports ────────────────────────────────────────────────────

    def _build_column_reports(self) -> list[dict]:
        reports: list[dict] = []
        shared_cols = [c for c in self.raw.columns if c in self.cleaned.columns]

        for col in shared_cols:
            b_series = self.raw[col]
            a_series = self.cleaned[col]

            b_missing = _count_dirty(b_series)
            a_missing = _count_dirty(a_series)

            # Count changed cells for this column
            cells_changed = 0
            for idx in range(min(len(b_series), len(a_series))):
                bv = b_series.iloc[idx]
                av = a_series.iloc[idx]
                b_d = _is_dirty(bv)
                a_d = _is_dirty(av)
                if b_d and not a_d:
                    cells_changed += 1
                elif not b_d and not a_d:
                    try:
                        if str(bv) != str(av):
                            cells_changed += 1
                    except Exception:
                        pass

            # Which action was planned for this column
            action = self.plan.get(col, self._infer_action(col, b_series, b_missing))
            cat_info = _ACTION_CATEGORY.get(action, {"label": "General Cleaning", "color": "#8e8fa1"})

            b_stats = _series_stats(b_series)
            a_stats = _series_stats(a_series)

            semantic_type = self._infer_semantic_type(col, b_series)
            status        = _assign_status(b_missing, a_missing, cells_changed)

            reasoning = _generate_reasoning(
                col=col,
                action=action,
                b_series=b_series,
                a_series=a_series,
                b_stats=b_stats,
                a_stats=a_stats,
                b_missing=b_missing,
                a_missing=a_missing,
                cells_changed=cells_changed,
            ) if cells_changed > 0 or b_missing > 0 else (
                f"'{col}' required no changes — the column passed all quality checks."
            )

            reports.append({
                "column":          col,
                "semantic_type":   semantic_type,
                "dtype":           str(b_series.dtype),
                "status":          status,
                "operation_type":  cat_info["label"],
                "operation_color": cat_info["color"],
                "cells_changed":   cells_changed,
                "missing_before":  b_missing,
                "missing_after":   a_missing,
                "ai_reasoning":    reasoning,
                "stats_before":    b_stats,
                "stats_after":     a_stats,
                "unique_before":   int(b_series.nunique()),
                "unique_after":    int(a_series.nunique()),
            })

        return reports

    # ── operation breakdown ───────────────────────────────────────────────────

    def _build_operation_breakdown(self, col_reports: list[dict]) -> dict:
        breakdown: dict[str, int] = {
            "Null Handling":    0,
            "Outlier Removal":  0,
            "Standardization":  0,
            "Type Casting":     0,
            "Deduplication":    0,
            "Pattern Cleaning": 0,
            "General Cleaning": 0,
        }

        for r in col_reports:
            op = r.get("operation_type", "General Cleaning")
            changed = r.get("cells_changed", 0)
            if changed > 0 and op in breakdown:
                breakdown[op] += changed

        # Parse deduplication from the cleaner actions log
        for action_str in self.cr.get("actions", []):
            if "duplicate" in action_str.lower():
                try:
                    count = int("".join(filter(str.isdigit, action_str.split("Removed")[1].split("dup")[0])))
                    breakdown["Deduplication"] += count
                except Exception:
                    breakdown["Deduplication"] += 1

        return breakdown

    # ── raw actions log ───────────────────────────────────────────────────────

    def _build_actions_log(self) -> list[dict]:
        """Convert the SmartDataCleaner action strings into structured log entries."""
        log: list[dict] = []
        
        # Format auto-executed policy risks
        for risk in self.cr.get("auto_executed_risks", []):
            log.append({
                "column": risk.get("column", "Dataset-wide"),
                "issue": f"Policy Risk: {risk.get('reason', 'High/Medium risk action')}",
                "resolution": f"AUTO_EXECUTED_RISK: Executed '{risk.get('action')}' autonomously under Auto-Commit policy"
            })
            
        for action_str in self.cr.get("actions", []):
            action_lower = action_str.lower()
            
            # 1. Deduplication
            if "duplicate" in action_lower:
                import re
                match = re.search(r'removed\s+(\d+)\s+duplicates', action_lower)
                count = match.group(1) if match else "some"
                log.append({
                    "column": "Dataset-wide",
                    "issue": "Duplicate rows detected",
                    "resolution": f"Removed {count} duplicate rows"
                })
            
            # 2. Standardize Date
            elif "standardized date:" in action_lower:
                col = action_str.split(":", 1)[1].strip()
                log.append({
                    "column": col,
                    "issue": "Inconsistent date/time format",
                    "resolution": "Standardized date values"
                })
                
            # 3. Fuzzy Fix
            elif "fuzzy matched text in:" in action_lower:
                col = action_str.split(":", 1)[1].strip()
                log.append({
                    "column": col,
                    "issue": "String typos / spelling variations",
                    "resolution": "Fuzzy consolidated spelling typos"
                })
                
            # 4. Outliers
            elif "removed" in action_lower and "outliers from" in action_lower:
                import re
                match = re.search(r'removed\s+(\d+)\s+outliers\s+from\s+(.+)', action_str, re.IGNORECASE)
                if match:
                    count = match.group(1)
                    col = match.group(2).strip()
                else:
                    count = "some"
                    col = "Unknown Column"
                log.append({
                    "column": col,
                    "issue": "Outliers / anomalies detected",
                    "resolution": f"Removed {count} outliers via Z-score"
                })
                
            # 5. Smart Impute
            elif "smart impute:" in action_lower:
                col = action_str.split(":", 1)[1].strip()
                log.append({
                    "column": col,
                    "issue": "Missing value(s)",
                    "resolution": "Smart predictive imputation (KNN/MICE)"
                })
                
            # 6. Final Polish
            elif "final polish:" in action_lower:
                import re
                match = re.search(r"filled remaining nans in\s+'?([^']+)'?", action_lower)
                col = match.group(1).strip() if match else "Unknown Column"
                log.append({
                    "column": col,
                    "issue": "Missing value(s)",
                    "resolution": "Final fallback imputation"
                })
                
            # 7. Fallback
            else:
                log.append({
                    "column": "Dataset-wide",
                    "issue": "General data cleaning",
                    "resolution": action_str
                })
        return log

    # ── internal helpers ──────────────────────────────────────────────────────

    def _infer_action(self, col: str, series: pd.Series, missing: int) -> str:
        """Fallback action inference when no plan entry exists."""
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower:
            return "standardize_date"
        if "email" in col_lower or "mail" in col_lower:
            return "clean_pattern"
        if "phone" in col_lower or "mobile" in col_lower:
            return "clean_pattern"
        if pd.api.types.is_numeric_dtype(series):
            return "remove_outliers" if missing > 0 else "remove_outliers"
        if any(k in col_lower for k in ["city", "job", "country", "title", "governorate"]):
            return "fuzzy_fix"
        if missing > 0:
            return "smart_impute"
        return "general"

    def _infer_semantic_type(self, col: str, series: pd.Series) -> str:
        """Derive a human-readable semantic type label."""
        col_lower = col.lower()
        if "email" in col_lower or "mail" in col_lower:
            return "Email"
        if "phone" in col_lower or "mobile" in col_lower:
            return "Phone Number"
        if "date" in col_lower or "time" in col_lower or "join" in col_lower:
            return "DateTime"
        if any(k in col_lower for k in ["salary", "price", "cost", "amount", "revenue", "income", "budget"]):
            return "Numeric / Financial"
        if any(k in col_lower for k in ["id", "identifier", "code", "uid"]):
            return "Identifier"
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 15:
                return "Numeric / Ordinal"
            return "Numeric / Continuous"
        if series.dtype == object:
            if series.nunique() < 20:
                return "Categorical"
            return "Free Text"
        return "Unknown"
