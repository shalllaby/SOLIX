import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

class SOLAutoMLVisualizer:
    """
    Intelligent Visualizer for SOL AutoML.
    Generates premium interactive Plotly figures for Streamlit,
    and handles double-fallback static PNG exporting using Matplotlib
    if Kaleido is not installed or errors out.
    """
    
    @staticmethod
    def save_figure(fig: Any, output_path: str, fallback_func: Any, fallback_args: tuple) -> bool:
        """
        Saves a Plotly figure as PNG. If Kaleido fails, falls back to the Matplotlib generator.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            # Try Plotly static export (Kaleido)
            fig.write_image(output_path, format="png", engine="kaleido")
            return True
        except Exception:
            # Fallback to Matplotlib rendering
            try:
                fallback_func(*fallback_args, output_path)
                return True
            except Exception as fallback_err:
                print(f"Fallback visualization save failed for {output_path}: {fallback_err}")
                return False

    # ------------------------------------------------
    # 1. Confusion Matrix
    # ------------------------------------------------
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> go.Figure:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix Analysis",
            x=labels,
            y=labels,
            labels=dict(x="Predicted Class", y="Actual Class")
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], output_path: str):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix Analysis')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        # Annotate
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
                         
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 2. ROC Curve
    # ------------------------------------------------
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, task_type: str) -> go.Figure:
        fig = go.Figure()
        
        if task_type == "binary":
            # Extract positive-class probabilities from 2D predict_proba output
            if y_probs.ndim == 2:
                y_scores = y_probs[:, 1]
            else:
                y_scores = y_probs
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#2C4A7F', width=2.5)))
        else:
            # Multiclass: draw One-vs-Rest for each class
            for i in range(y_probs.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Class {i}'))
                
        fig.add_shape(type='line', line=dict(dash='dash', color='red', width=1.5), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(
            title="ROC Curve Analysis",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, task_type: str, output_path: str):
        plt.figure(figsize=(6, 5))
        if task_type == "binary":
            # Extract positive-class probabilities from 2D predict_proba output
            if y_probs.ndim == 2:
                y_scores = y_probs[:, 1]
            else:
                y_scores = y_probs
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, color='#2C4A7F', lw=2, label='ROC Curve')
        else:
            for i in range(y_probs.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
                plt.plot(fpr, tpr, lw=1.5, label=f'Class {i}')
                
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Analysis')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 3. Precision-Recall Curve
    # ------------------------------------------------
    @staticmethod
    def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, task_type: str) -> go.Figure:
        fig = go.Figure()
        
        if task_type == "binary":
            # Extract positive-class probabilities from 2D predict_proba output
            if y_probs.ndim == 2:
                y_scores = y_probs[:, 1]
            else:
                y_scores = y_probs
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve', line=dict(color='#10B981', width=2.5)))
        else:
            for i in range(y_probs.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
                fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Class {i}'))
                
        fig.update_layout(
            title="Precision-Recall Curve Analysis",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, task_type: str, output_path: str):
        plt.figure(figsize=(6, 5))
        if task_type == "binary":
            # Extract positive-class probabilities from 2D predict_proba output
            if y_probs.ndim == 2:
                y_scores = y_probs[:, 1]
            else:
                y_scores = y_probs
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            plt.plot(recall, precision, color='#10B981', lw=2, label='PR Curve')
        else:
            for i in range(y_probs.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
                plt.plot(recall, precision, lw=1.5, label=f'Class {i}')
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Analysis')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 4. Feature Importance
    # ------------------------------------------------
    @staticmethod
    def plot_feature_importance(importance_data: Any) -> go.Figure:
        if isinstance(importance_data, dict):
            importance_data = [
                {"feature": f, "importance": float(i)}
                for f, i in importance_data.items()
            ]
        df_imp = pd.DataFrame(importance_data[:10]).sort_values(by="importance", ascending=True)
        fig = px.bar(
            df_imp,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Feature Importance Profile",
            color="importance",
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        return fig

    @staticmethod
    def _fallback_feature_importance(importance_data: Any, output_path: str):
        if isinstance(importance_data, dict):
            importance_data = [
                {"feature": f, "importance": float(i)}
                for f, i in importance_data.items()
            ]
        df_imp = pd.DataFrame(importance_data[:10]).sort_values(by="importance", ascending=True)
        plt.figure(figsize=(6, 5))
        plt.barh(df_imp["feature"], df_imp["importance"], color='#2C4A7F')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Feature Importance Profile')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 5. Class Target Distribution
    # ------------------------------------------------
    @staticmethod
    def plot_target_distribution(y: np.ndarray, labels: List[str]) -> go.Figure:
        unique, counts = np.unique(y, return_counts=True)
        df_dist = pd.DataFrame({"Class": [labels[int(u)] for u in unique], "Count": counts})
        fig = px.pie(
            df_dist,
            names="Class",
            values="Count",
            title="Target Class Ratios",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_target_distribution(y: np.ndarray, labels: List[str], output_path: str):
        unique, counts = np.unique(y, return_counts=True)
        class_names = [labels[int(u)] for u in unique]
        plt.figure(figsize=(6, 5))
        plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=140, colors=['#EFF6FF', '#BFDBFE', '#93C5FD', '#60A5FA'])
        plt.title('Target Class Ratios')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 6. Residual Plot (Regression)
    # ------------------------------------------------
    @staticmethod
    def plot_residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        residuals = y_true - y_pred
        fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={"x": "Predicted Target", "y": "Residual Value (Error)"},
            title="Residual Plot Analysis"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residuals, alpha=0.6, color='#2C4A7F')
        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.xlabel('Predicted Target')
        plt.ylabel('Residual Value (Error)')
        plt.title('Residual Plot Analysis')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 7. Prediction vs Actual (Regression)
    # ------------------------------------------------
    @staticmethod
    def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            labels={"x": "Actual Value", "y": "Predicted Value"},
            title="Prediction vs. Actual Comparison"
        )
        # Add 45-degree line
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Reference Line', line=dict(color='red', dash='dash')))
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.6, color='#2C4A7F')
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], color='red', linestyle='--', lw=2, label='Reference')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title('Prediction vs. Actual Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 8. Model Comparison Leaderboard
    # ------------------------------------------------
    @staticmethod
    def plot_leaderboard_comparison(leaderboard_df: pd.DataFrame, metric_name: str) -> go.Figure:
        df = leaderboard_df.copy()
        if metric_name not in df.columns and "val_metrics" in df.columns:
            df[metric_name] = df["val_metrics"].apply(lambda m: m.get(metric_name, 0.0) if isinstance(m, dict) else 0.0)
            
        fig = px.bar(
            df,
            x="model_name",
            y=["composite_score", metric_name],
            barmode="group",
            title="Model Metrics & Composite Score Leaderboard",
            color_discrete_sequence=["#2C4A7F", "#3B82F6"],
            labels={"value": "Performance Metric Scale", "model_name": "ML Algorithm"}
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def _fallback_leaderboard_comparison(leaderboard_df: pd.DataFrame, metric_name: str, output_path: str):
        df = leaderboard_df.copy()
        if metric_name not in df.columns and "val_metrics" in df.columns:
            df[metric_name] = df["val_metrics"].apply(lambda m: m.get(metric_name, 0.0) if isinstance(m, dict) else 0.0)
            
        plt.figure(figsize=(7, 5))
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df["composite_score"], width, label='Composite Score', color='#2C4A7F')
        plt.bar(x + width/2, df[metric_name], width, label=metric_name.upper(), color='#3B82F6')
        
        plt.xlabel('ML Algorithm')
        plt.ylabel('Performance Metric Scale')
        plt.title('Model Metrics & Composite Score Leaderboard')
        plt.xticks(x, df["model_name"], rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------
    # 9. Training Runtime Chart
    # ------------------------------------------------
    @staticmethod
    def plot_runtime_comparison(leaderboard_df: pd.DataFrame) -> go.Figure:
        fig = px.bar(
            leaderboard_df.sort_values(by="fit_time", ascending=True),
            x="fit_time",
            y="model_name",
            orientation="h",
            title="Algorithm Training Duration Runtimes",
            color="fit_time",
            color_continuous_scale="Teal",
            labels={"fit_time": "Fitting Time (Seconds)", "model_name": "ML Algorithm"}
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        return fig

    @staticmethod
    def _fallback_runtime_comparison(leaderboard_df: pd.DataFrame, output_path: str):
        df_sorted = leaderboard_df.sort_values(by="fit_time", ascending=True)
        plt.figure(figsize=(7, 5))
        plt.barh(df_sorted["model_name"], df_sorted["fit_time"], color='#14B8A6')
        plt.xlabel('Fitting Time (Seconds)')
        plt.ylabel('ML Algorithm')
        plt.title('Algorithm Training Duration Runtimes')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
