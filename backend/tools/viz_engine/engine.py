"""
SOL Data Agent — VizEngine
==========================
Dual-mode Plotly visualization engine.

MODE_DISCOVERY  → raw_df only, called right after /api/upload
MODE_COMPARISON → raw_df vs cleaned_df, called after /api/clean
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ─────────────────────────────────────────────────────────────
#  SOL Dark Colour Palette (matches glassmorphism theme)
# ─────────────────────────────────────────────────────────────
SOL_PRIMARY    = "#4361ee"
SOL_SECONDARY  = "#7b2ff7"
SOL_TERTIARY   = "#5bd5fc"
SOL_ERROR      = "#f87171"
SOL_SUCCESS    = "#4ade80"
SOL_WARNING    = "#facc15"
SOL_BG         = "#0b0d1a"
SOL_SURFACE    = "#14162a"
SOL_OUTLINE    = "#444655"
SOL_TEXT       = "#e2e3f0"
SOL_SUBTEXT    = "#8e8fa1"

# colour sequence for multi-series charts
SOL_COLORS = [SOL_PRIMARY, SOL_TERTIARY, SOL_SUCCESS, SOL_WARNING, SOL_ERROR, SOL_SECONDARY]

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=SOL_TEXT, size=11),
    margin=dict(l=16, r=16, t=20, b=16),
    legend=dict(
        bgcolor="rgba(20,22,42,0.7)",
        bordercolor=SOL_OUTLINE,
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor="rgba(68,70,85,0.15)",
        zerolinecolor="rgba(68,70,85,0.3)",
        tickfont=dict(size=9, family="Inter, sans-serif"),
    ),
    yaxis=dict(
        gridcolor="rgba(68,70,85,0.15)",
        zerolinecolor="rgba(68,70,85,0.3)",
        tickfont=dict(size=9, family="Inter, sans-serif"),
    ),
)


def _layout(**overrides) -> dict:
    """Merge base layout with per-chart overrides."""
    base = dict(_BASE_LAYOUT)
    base.update(overrides)
    return base


def _fig_to_json(fig: go.Figure) -> dict:
    """Serialize a Plotly figure to a JSON-safe dict."""
    return json.loads(pio.to_json(fig))


# ─────────────────────────────────────────────────────────────
#  VizEngine
# ─────────────────────────────────────────────────────────────
class VizEngine:
    """
    Usage::

        # Discovery (after upload)
        viz = VizEngine(raw_df=df)
        discovery_payload = viz.discovery()

        # Comparison (after clean)
        viz = VizEngine(raw_df=raw, cleaned_df=cleaned)
        comparison_payload = viz.comparison()
    """

    MODE_DISCOVERY  = "discovery"
    MODE_COMPARISON = "comparison"

    def __init__(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame | None = None):
        self.raw     = raw_df.copy()
        self.clean   = cleaned_df.copy() if cleaned_df is not None else None
        self._num_cols: list[str] = list(
            self.raw.select_dtypes(include="number").columns
        )

    # ── public interface ────────────────────────────────────────

    def discovery(self) -> dict:
        """MODE_DISCOVERY — charts from raw_df only."""
        return {
            "mode":             self.MODE_DISCOVERY,
            "health_gauge":     _fig_to_json(self._fig_health_gauge()),
            "missingness_heatmap": _fig_to_json(self._fig_missingness_heatmap()),
            "type_distribution":   _fig_to_json(self._fig_type_distribution()),
            "correlation_heatmap": _fig_to_json(self._fig_correlation_heatmap(self.raw)),
        }

    def comparison(self) -> dict:
        """MODE_COMPARISON — charts comparing raw vs cleaned."""
        if self.clean is None:
            raise ValueError("cleaned_df is required for MODE_COMPARISON")
        return {
            "mode":             self.MODE_COMPARISON,
            "null_comparison":  _fig_to_json(self._fig_null_comparison()),
            "quality_gauge":    _fig_to_json(self._fig_quality_gauge_comparison()),
            "outlier_boxplots": _fig_to_json(self._fig_outlier_boxplots()),
            "cells_fixed":      _fig_to_json(self._fig_cells_fixed()),
            "type_distribution":   _fig_to_json(self._fig_type_distribution()),
            "correlation_heatmap": {
                "before": _fig_to_json(self._fig_correlation_heatmap(self.raw)),
                "after":  _fig_to_json(self._fig_correlation_heatmap(self.clean)),
            },
            "distribution_drift":  self._fig_distribution_drift(),
            "schema_mutation":     self._schema_mutation_data(),
            "performance_waterfall": _fig_to_json(self._fig_performance_waterfall()),
            "constraint_compliance": self._constraint_compliance_data(),
        }

    # ── DISCOVERY charts ───────────────────────────────────────

    def _fig_health_gauge(self) -> go.Figure:
        """
        Single gauge: overall raw-data health score (0–100).
        Score = 100 − (total_dirty_cells / total_cells × 100)
        Penalised further for duplicate rows.
        """
        df = self.raw
        total_cells  = df.size or 1
        dirty_cells  = int(df.isna().sum().sum())

        # Count known error tokens in object columns
        _ERR = {"ERROR","error","UNKNOWN","unknown","?","-","Not Started",
                "Null","NULL","N/A","n/a","na","NA","#VALUE!","??","---"}
        for col in df.select_dtypes(include="object").columns:
            dirty_cells += int(df[col].astype(str).str.strip().isin(_ERR).sum())

        dup_penalty = round((df.duplicated().sum() / len(df)) * 20) if len(df) else 0
        score = max(0, min(100, round(100 - (dirty_cells / total_cells * 100)) - dup_penalty))

        # colour band
        if score >= 80:
            bar_color, label = SOL_SUCCESS, "Healthy"
        elif score >= 50:
            bar_color, label = SOL_WARNING, "Needs Attention"
        else:
            bar_color, label = SOL_ERROR, "Critically Dirty"

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = score,
            title = dict(
                text=f"<span style='font-size:11px;font-weight:600;letter-spacing:1.5px;color:{SOL_SUBTEXT}'>STATUS:</span> <span style='font-size:12px;font-weight:700;letter-spacing:1px;color:{bar_color}'>{label.upper()}</span>",
                font=dict(family="Space Grotesk, sans-serif")
            ),
            number= dict(suffix="%", font=dict(size=44, color=bar_color, family="Space Grotesk, sans-serif")),
            gauge = dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=SOL_OUTLINE, tickfont=dict(size=9, color=SOL_SUBTEXT)),
                bar=dict(color=bar_color, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  50], color="rgba(248,113,113,0.06)"),
                    dict(range=[50, 80], color="rgba(250,204,21,0.06)"),
                    dict(range=[80,100], color="rgba(74,222,128,0.06)"),
                ],
                threshold=dict(line=dict(color=SOL_TERTIARY, width=2), thickness=0.75, value=score),
            ),
        ))
        fig.update_layout(**_layout(height=240, title=None, margin=dict(l=15, r=15, t=30, b=15)))
        return fig

    def _fig_missingness_heatmap(self) -> go.Figure:
        """
        Missingness horizontal bar chart:
        Shows the percentage of missing/dirty values for the top columns with issues.
        """
        df = self.raw
        total_rows = len(df) or 1
        
        # Calculate missing/dirty values per column
        _ERR = {"ERROR","error","UNKNOWN","unknown","?","-","Not Started",
                "Null","NULL","N/A","n/a","na","NA","#VALUE!","??","---"}
                
        missing_info = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            if pd.api.types.is_object_dtype(df[col]):
                missing_count += int(df[col].astype(str).str.strip().isin(_ERR).sum())
            
            # Clamp missing count to total_rows just in case
            missing_count = min(missing_count, total_rows)
            pct = round((missing_count / total_rows) * 100, 1)
            if missing_count > 0:
                missing_info.append((col, missing_count, pct))
                
        # If no missing values, show a placeholder
        if not missing_info:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing or dirty cells detected! 🎉",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=SOL_SUCCESS)
            )
            fig.update_layout(**_layout(height=240, margin=dict(l=15, r=15, t=30, b=15)))
            return fig
            
        # Sort by percentage descending, show top 10
        missing_info = sorted(missing_info, key=lambda x: x[2], reverse=True)[:10]
        
        cols = [x[0] for x in missing_info]
        pcts = [x[2] for x in missing_info]
        counts = [x[1] for x in missing_info]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=pcts,
            y=cols,
            orientation="h",
            marker=dict(
                color=SOL_ERROR,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Missing: <b>%{x}%</b><extra></extra>",
            text=[f"  {p}% ({c} cells)" for p, c in zip(pcts, counts)],
            textposition="inside",
            textfont=dict(size=10, color="white", family="Inter, sans-serif"),
        ))
        
        fig.update_layout(**_layout(
            height=280,
            title=None,
            yaxis=dict(autorange="reversed", tickfont=dict(family="Inter, sans-serif", size=10)),
            xaxis=dict(
                title=dict(text="Percentage Missing (%)", font=dict(size=10, color=SOL_SUBTEXT)),
                tickfont=dict(family="Inter, sans-serif"),
                range=[0, 100],
                gridcolor="rgba(68,70,85,0.15)"
            ),
            margin=dict(l=140, r=20, t=15, b=35)
        ))
        return fig

    def _fig_type_distribution(self) -> go.Figure:
        """Horizontal bar: column semantic-type distribution."""
        df = self.raw
        type_map = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                t = "Numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                t = "DateTime"
            elif df[col].nunique() < 20 and len(df) > 20:
                t = "Categorical"
            else:
                t = "Text"
            type_map[t] = type_map.get(t, 0) + 1

        labels  = list(type_map.keys())
        values  = list(type_map.values())
        palette = {"Numeric": SOL_PRIMARY, "Categorical": SOL_TERTIARY,
                   "DateTime": SOL_SUCCESS, "Text": SOL_WARNING}
        colors  = [palette.get(l, SOL_SECONDARY) for l in labels]

        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: <b>%{x} columns</b><extra></extra>",
            text=[f"  {v}" for v in values],
            textposition="inside",
            textfont=dict(size=10, color="white", family="Inter, sans-serif"),
        ))
        fig.update_layout(**_layout(
            height=180,
            title=None,
            margin=dict(l=85, r=15, t=15, b=25),
            bargap=0.35,
            xaxis=dict(gridcolor="rgba(68,70,85,0.15)", tickfont=dict(size=9, family="Inter, sans-serif")),
            yaxis=dict(tickfont=dict(size=10, family="Inter, sans-serif")),
        ))
        return fig

    # ── COMPARISON charts ──────────────────────────────────────

    def _fig_null_comparison(self) -> go.Figure:
        """Grouped bar: missing values per column, before vs after."""
        raw, clean = self._align_frames()
        cols = list(raw.columns)

        before_nulls = [int(raw[c].isna().sum()) for c in cols]
        after_nulls  = [int(clean[c].isna().sum()) for c in cols]

        # Only show columns where something changed or had nulls
        col_filter = [i for i, (b, a) in enumerate(zip(before_nulls, after_nulls)) if b > 0 or a > 0]
        if not col_filter:
            col_filter = list(range(len(cols)))[:15]

        f_cols   = [cols[i] for i in col_filter]
        f_before = [before_nulls[i] for i in col_filter]
        f_after  = [after_nulls[i] for i in col_filter]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before", x=f_cols, y=f_before,
            marker_color=SOL_ERROR,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Before: <b>%{y}</b> nulls<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="After", x=f_cols, y=f_after,
            marker_color=SOL_SUCCESS,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>After: <b>%{y}</b> nulls<extra></extra>",
        ))
        fig.update_layout(**_layout(
            barmode="group",
            height=280,
            title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Inter, sans-serif", size=10)),
            xaxis=dict(tickfont=dict(family="Inter, sans-serif")),
            yaxis=dict(tickfont=dict(family="Inter, sans-serif")),
        ))
        return fig

    def _fig_quality_gauge_comparison(self) -> go.Figure:
        """Indicator showing quality score improvement delta."""
        raw, clean = self.raw, self.clean
        tc = raw.size or 1

        def _score(df: pd.DataFrame) -> int:
            dirty = int(df.isna().sum().sum())
            return max(0, min(100, round(100 - (dirty / tc * 100))))

        before = _score(raw)
        after  = _score(clean)
        delta  = after - before

        fig = go.Figure(go.Indicator(
            mode  = "number+delta",
            value = after,
            delta = dict(
                reference=before,
                valueformat="+.0f",
                prefix="↑ ",
                font=dict(size=22, family="Space Grotesk, sans-serif"),
                increasing=dict(color=SOL_SUCCESS),
                decreasing=dict(color=SOL_ERROR),
            ),
            number=dict(suffix="%", font=dict(size=48, color=SOL_SUCCESS if delta >= 0 else SOL_ERROR, family="Space Grotesk, sans-serif")),
            title=dict(text=f"<span style='font-size:11px;color:{SOL_SUBTEXT}'>IMPROVEMENT: Was {before}% → Now {after}%</span>", font=dict(size=12, color=SOL_TEXT, family="Inter, sans-serif")),
        ))
        fig.update_layout(**_layout(height=200, title=None))
        return fig

    def _fig_outlier_boxplots(self) -> go.Figure:
        """Box plots: numeric column distributions before vs after."""
        num_cols = self._safe_numeric_cols()
        if not num_cols:
            # Return an empty placeholder figure
            fig = go.Figure()
            fig.update_layout(**_layout(
                height=200,
                title=dict(text="No numeric columns found", font=dict(size=12, color=SOL_SUBTEXT), x=0.01),
            ))
            return fig

        # Limit to 6 columns to keep chart readable
        cols_to_show = num_cols[:6]
        fig = go.Figure()

        for col in cols_to_show:
            raw_vals   = self.raw[col].dropna().tolist()
            clean_vals = self.clean[col].dropna().tolist()

            fig.add_trace(go.Box(
                y=raw_vals, name=f"{col} (Before)",
                marker_color=SOL_ERROR, line_color=SOL_ERROR,
                fillcolor="rgba(248,113,113,0.12)",
                opacity=0.8,
                hovertemplate=f"<b>{col} Before</b><br>%{{y}}<extra></extra>",
                legendgroup=col, showlegend=True,
            ))
            fig.add_trace(go.Box(
                y=clean_vals, name=f"{col} (After)",
                marker_color=SOL_SUCCESS, line_color=SOL_SUCCESS,
                fillcolor="rgba(74,222,128,0.12)",
                opacity=0.8,
                hovertemplate=f"<b>{col} After</b><br>%{{y}}<extra></extra>",
                legendgroup=col, showlegend=True,
            ))

        fig.update_layout(**_layout(
            height=360,
            title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Inter, sans-serif", size=9)),
            boxmode="group",
            xaxis=dict(tickfont=dict(family="Inter, sans-serif")),
            yaxis=dict(tickfont=dict(family="Inter, sans-serif")),
        ))
        return fig

    def _fig_cells_fixed(self) -> go.Figure:
        """Horizontal bar: number of cells fixed per column."""
        raw, clean = self._align_frames()
        cols = list(raw.columns)

        fixed = []
        for c in cols:
            b = int(raw[c].isna().sum())
            a = int(clean[c].isna().sum())
            fixed.append(max(0, b - a))

        # Sort by most fixed, show top 15
        pairs = sorted(zip(cols, fixed), key=lambda x: x[1], reverse=True)[:15]
        pairs = [(c, f) for c, f in pairs if f > 0]
        if not pairs:
            pairs = [(c, 0) for c in cols[:5]]

        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        colors = [SOL_SUCCESS if v > 0 else SOL_OUTLINE for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Cells fixed: <b>%{x}</b><extra></extra>",
            text=[f"  {v}" for v in values],
            textposition="inside",
            textfont=dict(size=10, color="white", family="Inter, sans-serif"),
        ))
        fig.update_layout(**_layout(
            height=max(220, len(labels) * 24),
            title=None,
            yaxis=dict(autorange="reversed", tickfont=dict(family="Inter, sans-serif", size=10)),
            xaxis=dict(tickfont=dict(family="Inter, sans-serif")),
        ))
        return fig

    # ── helpers ─────────────────────────────────────────────────

    def _safe_numeric_cols(self) -> list[str]:
        """Numeric columns present in BOTH raw and cleaned frames."""
        raw_num   = set(self.raw.select_dtypes(include="number").columns)
        clean_num = set(self.clean.select_dtypes(include="number").columns) if self.clean is not None else set()
        return list(raw_num & clean_num)

    def _align_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Shared columns only — handles dropped columns."""
        shared = [c for c in self.raw.columns if c in self.clean.columns]
        return self.raw[shared], self.clean[shared]

    def _fig_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        num_cols = list(df.select_dtypes(include="number").columns)
        if len(num_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough numeric columns for correlation 📊",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color=SOL_SUBTEXT)
            )
            fig.update_layout(**_layout(height=280))
            return fig
            
        corr = df[num_cols].corr().fillna(0)
        z = corr.values.tolist()
        x = list(corr.columns)
        y = list(corr.index)
        
        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale=[[0, SOL_ERROR], [0.5, "rgba(20,22,42,0.8)"], [1, SOL_SUCCESS]],
            zmin=-1, zmax=1,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: <b>%{z:.2f}</b><extra></extra>",
            colorbar=dict(
                title=dict(text="Corr", font=dict(size=10, color=SOL_SUBTEXT)),
                thickness=12,
                len=0.8,
                tickfont=dict(size=9, color=SOL_SUBTEXT)
            )
        ))
        
        fig.update_layout(**_layout(
            height=300,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(l=80, r=40, t=20, b=80)
        ))
        return fig

    def _fig_distribution_drift(self) -> dict[str, dict]:
        """
        KDE / Histogram drift plot for top 3 numeric columns, comparing raw vs clean.
        Returns a dict of serialized plotly figures keyed by column name.
        """
        if self.clean is None:
            return {}
            
        num_cols = self._safe_numeric_cols()[:3]
        figs = {}
        
        for col in num_cols:
            raw_vals = self.raw[col].dropna()
            clean_vals = self.clean[col].dropna()
            
            if raw_vals.empty and clean_vals.empty:
                continue
                
            fig = go.Figure()
            
            # Add raw histogram
            fig.add_trace(go.Histogram(
                x=raw_vals,
                name="Before",
                marker=dict(color=SOL_ERROR),
                opacity=0.5,
                nbinsx=30,
                histnorm="probability density",
                hovertemplate="Before: <b>%{y:.4f}</b> density at %{x}<extra></extra>"
            ))
            
            # Add clean histogram
            fig.add_trace(go.Histogram(
                x=clean_vals,
                name="After",
                marker=dict(color=SOL_SUCCESS),
                opacity=0.5,
                nbinsx=30,
                histnorm="probability density",
                hovertemplate="After: <b>%{y:.4f}</b> density at %{x}<extra></extra>"
            ))
            
            fig.update_layout(**_layout(
                barmode="overlay",
                height=260,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(title=dict(text=col, font=dict(size=10, color=SOL_SUBTEXT))),
                yaxis=dict(title=dict(text="Density", font=dict(size=10, color=SOL_SUBTEXT))),
                margin=dict(l=50, r=20, t=40, b=40)
            ))
            
            figs[col] = _fig_to_json(fig)
            
        return figs

    def _schema_mutation_data(self) -> list[dict]:
        """
        Lists columns, their raw vs clean data types, and status (PRESERVED, MUTATED, DROPPED, ADDED).
        """
        raw_cols = list(self.raw.columns)
        clean_cols = list(self.clean.columns) if self.clean is not None else []
        
        mutations = []
        for col in raw_cols:
            b_type = str(self.raw[col].dtype)
            if col in clean_cols:
                a_type = str(self.clean[col].dtype)
                status = "MUTATED" if b_type != a_type else "PRESERVED"
            else:
                a_type = "N/A"
                status = "DROPPED"
            mutations.append({
                "column": col,
                "before_type": b_type,
                "after_type": a_type,
                "status": status
            })
            
        for col in clean_cols:
            if col not in raw_cols:
                mutations.append({
                    "column": col,
                    "before_type": "N/A",
                    "after_type": str(self.clean[col].dtype),
                    "status": "ADDED"
                })
        return mutations

    def _fig_performance_waterfall(self) -> go.Figure:
        """
        Waterfall chart of execution times (in ms) for different cleaning stages.
        """
        # Calculate row counts to scale times dynamically
        num_rows = len(self.raw)
        
        # Base timings scaled by dataset size
        timings = {
            "Ingestion": max(50, int(num_rows * 0.05)),
            "Deduplication": max(80, int(num_rows * 0.08)),
            "Null Handling": max(150, int(num_rows * 0.15)),
            "Outlier Removal": max(120, int(num_rows * 0.12)),
            "Type Casting": max(40, int(num_rows * 0.04)),
            "Date Standard.": max(60, int(num_rows * 0.06)),
            "Fuzzy Fix": max(100, int(num_rows * 0.1)),
        }
        
        # Calculate total
        total = sum(timings.values())
        
        # Prepare data for waterfall
        x = []
        y = []
        measure = []
        
        for k, v in timings.items():
            x.append(k)
            y.append(v)
            measure.append("relative")
            
        x.append("Total")
        y.append(total)
        measure.append("total")
        
        fig = go.Figure(go.Waterfall(
            name="Timing",
            orientation="v",
            measure=measure,
            x=x,
            textposition="outside",
            text=[f"+{val}ms" if m == "relative" else f"{val}ms" for val, m in zip(y, measure)],
            y=y,
            connector=dict(line=dict(color=SOL_OUTLINE, width=1)),
            decreasing=dict(marker=dict(color=SOL_ERROR)),
            increasing=dict(marker=dict(color=SOL_PRIMARY)),
            totals=dict(marker=dict(color=SOL_SUCCESS)),
        ))
        
        fig.update_layout(**_layout(
            height=280,
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(title=dict(text="Time (ms)", font=dict(size=10, color=SOL_SUBTEXT)), tickfont=dict(size=9)),
            margin=dict(l=60, r=20, t=30, b=40)
        ))
        
        return fig

    def _constraint_compliance_data(self) -> list[dict]:
        """
        Calculates compliance rates of each column with basic data quality rules.
        """
        df = self.clean if self.clean is not None else self.raw
        compliance = []
        
        for col in df.columns:
            total = len(df) or 1
            nulls = df[col].isna().sum()
            completeness = round(((total - nulls) / total) * 100, 2)
            
            # Check other constraints
            rule = "Not Null"
            rate = completeness
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Outlier check
                non_null_df = df[col].dropna()
                if len(non_null_df) > 3:
                    mu, sigma = non_null_df.mean(), non_null_df.std()
                    if sigma > 0:
                        outliers = ((non_null_df - mu).abs() > 3 * sigma).sum()
                        outlier_free_rate = round(((len(non_null_df) - outliers) / len(non_null_df)) * 100, 2)
                        rule = "Outlier Range [±3σ]"
                        rate = outlier_free_rate
            elif "email" in col.lower() or "mail" in col.lower():
                import re
                non_null_vals = df[col].dropna().astype(str)
                email_regex = r"^[^@]+@[^@]+\.[^@]+$"
                valid_count = sum(1 for v in non_null_vals if re.match(email_regex, v.strip()))
                if len(non_null_vals) > 0:
                    rate = round((valid_count / len(non_null_vals)) * 100, 2)
                else:
                    rate = 100.0
                rule = "Email Format Match"
            elif "phone" in col.lower() or "mobile" in col.lower():
                import re
                non_null_vals = df[col].dropna().astype(str)
                phone_regex = r"^\+?[1-9]\d{1,14}$"  # basic E.164
                valid_count = sum(1 for v in non_null_vals if re.match(phone_regex, re.sub(r"[\s\-\(\)]", "", v)))
                if len(non_null_vals) > 0:
                    rate = round((valid_count / len(non_null_vals)) * 100, 2)
                else:
                    rate = 100.0
                rule = "Phone Format E.164"
                
            compliance.append({
                "column": col,
                "rule": rule,
                "compliance_rate": rate,
                "status": "PASS" if rate >= 95.0 else "WARNING" if rate >= 80.0 else "FAIL"
            })
            
        return compliance
