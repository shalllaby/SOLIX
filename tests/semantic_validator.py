import numpy as np
import pandas as pd
from typing import Dict, Any, List

class SemanticValidator:
    """
    Validates logical bounds, datatype consistency, identifier preservation,
    and distribution drift (Wasserstein distance, variance collapse, quantile shifts).
    """
    def __init__(self):
        pass

    def _wasserstein_distance_1d(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes 1D Wasserstein Distance (Earth Mover's Distance) using pure NumPy.
        """
        if len(u) == 0 or len(v) == 0:
            return 0.0
        u_sorted = np.sort(u)
        v_sorted = np.sort(v)
        u_quantiles = np.linspace(0, 1, len(u))
        v_quantiles = np.linspace(0, 1, len(v))
        all_quantiles = np.unique(np.concatenate([u_quantiles, v_quantiles]))
        u_vals = np.interp(all_quantiles, u_quantiles, u_sorted)
        v_vals = np.interp(all_quantiles, v_quantiles, v_sorted)
        deltas = np.diff(all_quantiles)
        mean_diffs = np.abs(u_vals[:-1] - v_vals[:-1])
        return float(np.sum(mean_diffs * deltas))

    def validate_bounds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates logical rules (e.g. age between 0 and 120, positive salaries).
        Returns a dictionary listing row indices of failures.
        """
        failures = {
            "age_bounds": [],
            "salary_bounds": []
        }
        
        # Age bounds check
        if "age" in df.columns:
            age_col = df["age"].dropna()
            bad_age = age_col[(age_col < 0) | (age_col > 120)]
            failures["age_bounds"] = bad_age.index.tolist()

        # Salary bounds check
        if "salary" in df.columns:
            # We assume numeric salary
            sal_col = pd.to_numeric(df["salary"], errors='coerce').dropna()
            bad_sal = sal_col[sal_col < 0]
            failures["salary_bounds"] = bad_sal.index.tolist()

        return failures

    def validate_identifiers(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Checks that sensitive ID columns were not mutated or dropped.
        """
        id_cols = ["user_id", "tx_hash"]
        status = {col: "OK" for col in id_cols}
        details = {col: "" for col in id_cols}

        for col in id_cols:
            if col not in cleaned_df.columns:
                status[col] = "DROPPED"
                details[col] = f"Column '{col}' was deleted from the dataset."
                continue
            
            # Check row count match
            common_idx = raw_df.index.intersection(cleaned_df.index)
            if len(common_idx) == 0:
                status[col] = "INDEX_CORRUPTED"
                details[col] = "No common indices found between raw and cleaned."
                continue
                
            raw_vals = raw_df.loc[common_idx, col].astype(str)
            cleaned_vals = cleaned_df.loc[common_idx, col].astype(str)
            
            mutations = (raw_vals != cleaned_vals).sum()
            if mutations > 0:
                status[col] = "MUTATED"
                details[col] = f"{mutations} cells in identifier '{col}' were changed."
                
        return {"status": status, "details": details}

    def analyze_distribution_drift(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs detailed distribution drift checks:
        - Wasserstein distance
        - Quantile deviation (median shift)
        - Variance collapse
        """
        drift_report = {}
        numeric_cols = ["age", "salary"]

        for col in numeric_cols:
            if col not in raw_df.columns or col not in cleaned_df.columns:
                continue

            raw_vals = pd.to_numeric(raw_df[col], errors='coerce').dropna().values
            cleaned_vals = pd.to_numeric(cleaned_df[col], errors='coerce').dropna().values

            if len(raw_vals) == 0 or len(cleaned_vals) == 0:
                continue

            raw_std = np.std(raw_vals)
            raw_mean = np.mean(raw_vals)
            
            # 1. Wasserstein distance
            wd = self._wasserstein_distance_1d(raw_vals, cleaned_vals)
            normalized_wd = wd / raw_std if raw_std > 0 else 0.0

            # 2. Quantile deviation (Median shift)
            raw_median = np.median(raw_vals)
            cleaned_median = np.median(cleaned_vals)
            median_shift = abs(raw_median - cleaned_median) / raw_median if raw_median > 0 else 0.0

            # 3. Variance collapse check
            raw_var = np.var(raw_vals)
            cleaned_var = np.var(cleaned_vals)
            variance_collapse = (raw_var - cleaned_var) / raw_var if raw_var > 0 else 0.0

            drift_report[col] = {
                "wasserstein_distance": wd,
                "normalized_wd": normalized_wd,
                "median_shift_pct": median_shift * 100,
                "variance_collapse_pct": variance_collapse * 100,
                "alert": normalized_wd > 0.25 or variance_collapse > 0.50
            }

        return drift_report

    def analyze_correlation_preservation(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Measures the absolute shift in Pearson correlation between numeric columns.
        """
        cols = ["age", "salary"]
        # Ensure they are numeric
        r_df = raw_df[cols].apply(pd.to_numeric, errors='coerce').dropna()
        c_df = cleaned_df[cols].apply(pd.to_numeric, errors='coerce').dropna()

        if len(r_df) < 5 or len(c_df) < 5:
            return {"correlation_shift": 0.0, "status": "INSUFFICIENT_DATA"}

        raw_corr = r_df.corr().iloc[0, 1]
        cleaned_corr = c_df.corr().iloc[0, 1]

        shift = abs(raw_corr - cleaned_corr) if not (np.isnan(raw_corr) or np.isnan(cleaned_corr)) else 0.0
        return {
            "raw_correlation": raw_corr,
            "cleaned_correlation": cleaned_corr,
            "correlation_shift": shift,
            "status": "OK" if shift < 0.2 else "DRIFT_DETECTED"
        }

    def compute_scores(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes the final semantic metrics:
        - Semantic Integrity Score (SIS)
        - Realism Preservation Score (RPS)
        """
        # 1. Semantic Integrity Score (SIS)
        failures = self.validate_bounds(cleaned_df)
        total_rows = len(cleaned_df)
        
        # Row is failed if age bounds or salary bounds failed
        failed_indices = set(failures["age_bounds"] + failures["salary_bounds"])
        sis = ((total_rows - len(failed_indices)) / total_rows) * 100 if total_rows > 0 else 100.0

        # 2. Realism Preservation Score (RPS)
        drift = self.analyze_distribution_drift(raw_df, cleaned_df)
        corr_pres = self.analyze_correlation_preservation(raw_df, cleaned_df)

        # Penalties start at 100
        rps = 100.0
        for col, metrics in drift.items():
            if metrics["alert"]:
                rps -= 10.0
            if metrics["normalized_wd"] > 0.5:
                rps -= 15.0
                
        if corr_pres.get("correlation_shift", 0.0) > 0.2:
            rps -= 10.0

        return {
            "semantic_integrity_score": max(0.0, sis),
            "realism_preservation_score": max(0.0, rps),
            "bounds_failures": failures,
            "distribution_drift": drift,
            "correlation": corr_pres
        }
