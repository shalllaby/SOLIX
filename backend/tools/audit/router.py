"""
SOL Data Agent — Audit Report FastAPI Router
=============================================
Endpoints:
  GET  /api/audit/{dataset_id}              → Full structured AuditLog JSON
  GET  /api/audit/{dataset_id}/export       → Downloadable CSV or JSON certificate
  GET  /api/audit/{dataset_id}/diff         → Compact cell-diff map for the diff viewer
"""

from __future__ import annotations

import io
import json
import csv

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from backend.store import _audit_store
from backend.auth import get_current_user

router = APIRouter(prefix="/api/audit", tags=["Audit Report"])


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_log(dataset_id: str) -> dict:
    """Retrieve an audit log or raise 404."""
    log = _audit_store.get(dataset_id)
    if not log:
        raise HTTPException(
            status_code=404,
            detail=(
                "Audit log not found for this dataset. "
                "Make sure the dataset has been cleaned first via /api/clean."
            ),
        )
    return log


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint 1 — Full AuditLog
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}", summary="Get full audit report for a cleaned dataset")
async def get_audit_report(
    dataset_id: str,
    current_user=Depends(get_current_user),
):
    """
    Returns the complete structured AuditLog produced by AuditReportBuilder.

    Includes:
    - Quality scores (before / after / delta)
    - Global statistics (cells changed, duplicates removed, etc.)
    - Operation breakdown (by category)
    - Per-column intelligence reports with AI reasoning
    - Raw actions log from the SmartDataCleaner
    """
    log = _get_log(dataset_id)
    return JSONResponse(log)


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint 2 — Export (Data Cleaning Certificate)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/export", summary="Download audit report as CSV or JSON certificate")
async def export_audit_report(
    dataset_id: str,
    fmt: str = Query(default="json", description="Export format: 'csv' or 'json'"),
    current_user=Depends(get_current_user),
):
    """
    Generates a downloadable Data Cleaning Certificate.

    - `fmt=json` → Full nested AuditLog as a JSON file
    - `fmt=csv`  → Flattened per-column report as a CSV spreadsheet
    """
    log = _get_log(dataset_id)
    filename = log.get("filename", "dataset").replace(" ", "_")
    base_name = filename.rsplit(".", 1)[0]

    fmt = fmt.lower().strip()

    # ── JSON export ───────────────────────────────────────────────────────────
    if fmt == "json":
        content = json.dumps(log, indent=2, ensure_ascii=False)
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{base_name}_audit_certificate.json"'
            },
        )

    # ── CSV export ────────────────────────────────────────────────────────────
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        # ── Section 1: Certificate Header ──────────────────────────────────
        writer.writerow(["=== SOL Data Agent — Data Cleaning Certificate ==="])
        writer.writerow(["Dataset",        log.get("filename", "")])
        writer.writerow(["Dataset ID",     log.get("dataset_id", "")])
        writer.writerow(["Audit ID",       log.get("audit_id", "")])
        writer.writerow(["Generated At",   log.get("generated_at", "")])
        writer.writerow(["Strategy Used",  log.get("strategy_used", "").upper()])
        writer.writerow(["User Goal",      log.get("user_goal", "N/A")])
        writer.writerow([])

        # ── Section 2: Quality Scores ───────────────────────────────────────
        writer.writerow(["=== QUALITY SCORES ==="])
        writer.writerow(["Metric", "Before", "After", "Delta"])
        qs_before = log.get("quality_scores", {}).get("before", {})
        qs_after  = log.get("quality_scores", {}).get("after", {})
        qs_delta  = log.get("quality_scores", {}).get("delta", {})
        for key in ["overall", "completeness", "outlier_safety", "consistency"]:
            writer.writerow([
                key.replace("_", " ").title(),
                f"{qs_before.get(key, 0)}%",
                f"{qs_after.get(key, 0)}%",
                f"{qs_delta.get(key, 0):+d}%",
            ])
        writer.writerow([])

        # ── Section 3: Global Statistics ────────────────────────────────────
        writer.writerow(["=== GLOBAL STATISTICS ==="])
        gs = log.get("global_stats", {})
        for k, v in gs.items():
            writer.writerow([k.replace("_", " ").title(), v])
        writer.writerow([])

        # ── Section 4: Operation Breakdown ──────────────────────────────────
        writer.writerow(["=== OPERATION BREAKDOWN (cells changed per type) ==="])
        writer.writerow(["Operation Type", "Cells Changed"])
        for op, count in log.get("operation_breakdown", {}).items():
            writer.writerow([op, count])
        writer.writerow([])

        # ── Section 5: Per-Column Intelligence ──────────────────────────────
        writer.writerow(["=== COLUMN INTELLIGENCE REPORT ==="])
        writer.writerow([
            "Column", "Semantic Type", "Data Type", "Status",
            "Operation", "Cells Changed",
            "Missing Before", "Missing After",
            "Unique Before", "Unique After",
            "Mean Before", "Mean After",
            "Std Before", "Std After",
            "Min Before", "Min After",
            "Max Before", "Max After",
            "AI Reasoning",
        ])

        for r in log.get("column_reports", []):
            sb = r.get("stats_before") or {}
            sa = r.get("stats_after") or {}
            writer.writerow([
                r.get("column", ""),
                r.get("semantic_type", ""),
                r.get("dtype", ""),
                r.get("status", ""),
                r.get("operation_type", ""),
                r.get("cells_changed", 0),
                r.get("missing_before", 0),
                r.get("missing_after", 0),
                r.get("unique_before", ""),
                r.get("unique_after", ""),
                sb.get("mean", ""),
                sa.get("mean", ""),
                sb.get("std", ""),
                sa.get("std", ""),
                sb.get("min", ""),
                sa.get("min", ""),
                sb.get("max", ""),
                sa.get("max", ""),
                r.get("ai_reasoning", ""),
            ])

        writer.writerow([])

        # ── Section 6: Actions Log ───────────────────────────────────────────
        writer.writerow(["=== CLEANING ACTIONS LOG ==="])
        writer.writerow(["Step", "Category", "Action"])
        for entry in log.get("actions_log", []):
            writer.writerow([entry.get("step", ""), entry.get("category", ""), entry.get("action", "")])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue().encode("utf-8-sig")]),   # BOM for Excel compat
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{base_name}_audit_certificate.csv"'
            },
        )

    raise HTTPException(status_code=400, detail="fmt must be 'csv' or 'json'.")


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint 3 — Compact diff map (for the Cleaning Studio diff viewer)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{dataset_id}/diff", summary="Get compact cell-diff map")
async def get_diff_map(
    dataset_id: str,
    current_user=Depends(get_current_user),
):
    """
    Returns a compact summary of which columns changed and by how many cells.
    Lighter than the full audit report — intended for the Cleaning Studio diff panel.

    Response shape::

        {
          "dataset_id": "...",
          "diff_summary": [
            { "column": "Salary", "cells_changed": 127, "operation": "Outlier Removal" },
            ...
          ]
        }
    """
    log = _get_log(dataset_id)

    diff_summary = [
        {
            "column":        r["column"],
            "cells_changed": r["cells_changed"],
            "operation":     r["operation_type"],
            "status":        r["status"],
        }
        for r in log.get("column_reports", [])
        if r.get("cells_changed", 0) > 0
    ]

    return JSONResponse({
        "dataset_id":   dataset_id,
        "diff_summary": diff_summary,
        "total_columns_changed": len(diff_summary),
    })
