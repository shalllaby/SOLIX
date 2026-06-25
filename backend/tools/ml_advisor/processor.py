import pandas as pd
import numpy as np
import json
import os
import urllib.request
import urllib.error
import warnings
from typing import List, Dict, Any, Optional
from scipy import stats
from groq import AsyncGroq

warnings.filterwarnings("ignore")

class MLAdvisorProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.model = "llama-3.3-70b-versatile"

    def profile_dataset(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Extract a rich metadata packet from *df* relative to *target_column*.
        Adopted from the 'منه' high-performance engine.
        """
        if df.empty:
            return {}

        # If no target column provided, assume the last one
        if target_column is None or target_column not in df.columns:
            target_column = df.columns[-1]

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
        overall_missing_pct = round(float(df.isnull().values.mean() * 100), 4)

        skewness: dict[str, float] = {}
        for col in numerical_cols[:20]:
            col_data = df[col].dropna()
            if len(col_data) > 2:
                try:
                    skew_val = float(stats.skew(col_data))
                    skewness[col] = round(skew_val, 4)
                except Exception:
                    skewness[col] = None

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
            "target_column": target_column
        }

    async def get_recommendations(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Uses LLM (Groq Llama-3.3-70b) to reason about the best ML models using 'منه' logic."""
        
        resolved_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key or resolved_key.strip() == "" or resolved_key == "your_api_key_here":
            raise ValueError("Groq API Key is not configured. Please set it in Settings.")

        client = AsyncGroq(api_key=resolved_key)

        system_prompt = """You are an expert Machine Learning Solutions Architect. Your task is to provide 
a high-level strategic recommendation for building a predictive model based on 
the provided dataset metadata.

Analyze the following constraints:
1. Data Volume: Assess if the dataset size requires high-bias/low-variance models 
or can support high-complexity ensembles.
2. Feature Mixture: Evaluate the ratio of categorical vs. numerical features to 
suggest models with native handling of specific types.
3. Target Nature: Determine the exact ML task (Binary/Multi-class Classification, 
Regression, or Clustering).

Your Output Requirements:
- Provide EXACTLY 3 model recommendations.
- Score each model on a scale of 0 to 100 based on its suitability and expected performance for the given dataset.
- Respond ONLY with a strict JSON object — no markdown, no code fences, no prose.

Output Format (strict JSON):
{
  "task_identified": "...",
  "recommendations": [
    {
      "model": "...",
      "score": 95, 
      "pros": "...",
      "cons": "...",
      "reasoning": "..."
    }
  ]
}
"""
        user_message = f"Please analyse the following dataset metadata packet and provide your top-3 ML model recommendations in the required JSON format.\n\n```json\n{json.dumps(profile, indent=2)}\n```"

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(response, module_name="advisor")
            except Exception as e_log:
                print(f"Failed to log advisor token usage: {e_log}")
                
            content = response.choices[0].message.content or "{}"
            
            # Parse JSON result
            data = json.loads(content)
            if isinstance(data, dict) and "recommendations" in data:
                recs = data["recommendations"]
                # Add reasoning if missing (fallback for UI compatibility)
                for r in recs:
                    if "reasoning" not in r:
                        r["reasoning"] = r.get("pros", "Strategic fit for dataset topology.")
                return recs
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            print(f"ML Advisor API Error: {e}")
            raise e
