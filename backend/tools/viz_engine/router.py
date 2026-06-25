"""
SOL Data Agent — VizEngine FastAPI Router
=========================================
Endpoints:
  GET /api/viz/{dataset_id}/discovery    → 3 discovery charts (raw data only)
  GET /api/viz/{dataset_id}/comparison   → 5 comparison charts (raw vs cleaned)
  GET /api/viz/{dataset_id}/figure/{name} → Single named figure (lazy load)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from backend.store import _store, _viz_store, _discovery_store
from backend.auth import get_current_user

router = APIRouter(prefix="/api/viz", tags=["Visual Analytics"])


def _get_store(dataset_id: str, store: dict, label: str) -> dict:
    data = store.get(dataset_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"{label} not found for dataset '{dataset_id}'. "
                   f"Make sure the dataset has been {'uploaded' if label == 'Discovery data' else 'cleaned'} first.",
        )
    return data


# ─────────────────────────────────────────────────────────
#  Endpoint 1 — Discovery figures (after upload)
# ─────────────────────────────────────────────────────────
@router.get("/{dataset_id}/discovery", summary="Get discovery charts for raw data")
async def get_discovery_viz(
    dataset_id: str,
    current_user=Depends(get_current_user),
):
    """
    Returns 3 Plotly figures computed from raw data only.

    Triggered right after /api/upload succeeds.
    Response::

        {
          "mode": "discovery",
          "health_gauge":          { plotly figure JSON },
          "missingness_heatmap":   { plotly figure JSON },
          "type_distribution":     { plotly figure JSON }
        }
    """
    data = _get_store(dataset_id, _discovery_store, "Discovery data")
    return JSONResponse(data)


# ─────────────────────────────────────────────────────────
#  Endpoint 2 — Comparison figures (after clean)
# ─────────────────────────────────────────────────────────
@router.get("/{dataset_id}/comparison", summary="Get comparison charts: raw vs cleaned")
async def get_comparison_viz(
    dataset_id: str,
    current_user=Depends(get_current_user),
):
    """
    Returns 5 Plotly figures comparing raw vs cleaned data.

    Triggered after /api/clean succeeds.
    Response::

        {
          "mode": "comparison",
          "null_comparison":    { plotly figure JSON },
          "quality_gauge":      { plotly figure JSON },
          "outlier_boxplots":   { plotly figure JSON },
          "cells_fixed":        { plotly figure JSON },
          "type_distribution":  { plotly figure JSON }
        }
    """
    data = _get_store(dataset_id, _viz_store, "Comparison data")
    return JSONResponse(data)


# ─────────────────────────────────────────────────────────
#  Endpoint 3 — Single named figure (lazy load)
# ─────────────────────────────────────────────────────────
@router.get("/{dataset_id}/figure/{fig_name}", summary="Get a single named figure")
async def get_single_figure(
    dataset_id: str,
    fig_name: str,
    current_user=Depends(get_current_user),
):
    """
    Returns a single named Plotly figure from either store.
    Searches discovery_store first, then viz_store.
    """
    payload = _discovery_store.get(dataset_id, {})
    if fig_name not in payload:
        payload = _viz_store.get(dataset_id, {})
    figure = payload.get(fig_name)
    if not figure:
        raise HTTPException(
            status_code=404,
            detail=f"Figure '{fig_name}' not found for dataset '{dataset_id}'.",
        )
    return JSONResponse({"figure": figure})
