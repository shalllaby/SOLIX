import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MissingnessIntelligence:
    """
    Missingness Intelligence Engine:
    - Profiles missing data patterns.
    - Classifies missingness into MCAR (Missing Completely At Random),
      MAR (Missing At Random), or MNAR (Missing Not At Random).
    - Blocks imputation on MNAR columns where NULL holds semantic business meaning.
    """
    @staticmethod
    def analyze_missingness(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Analyzes the missingness pattern of a target column.
        Returns a dictionary containing missingness type and safety action.
        """
        missing_count = df[col].isna().sum()
        total_rows = len(df)
        
        if missing_count == 0:
            return {"type": "None", "should_impute": True, "reason": "No missing values."}

        missing_pct = (missing_count / total_rows) * 100
        col_lower = col.lower()

        # 1. Semantically Meaningful NULLs (MNAR Protected Registry)
        # E.g., 'exit_date' -> NaN means still active. 'termination_reason' -> NaN means not terminated.
        meaningful_null_keywords = ['exit', 'terminate', 'resign', 'end_date', 'leave', 'spouse', 'children']
        if any(keyword in col_lower for keyword in meaningful_null_keywords):
            return {
                "type": "MNAR (Meaningful NULLs)",
                "should_impute": False,
                "reason": f"Column '{col}' holds semantically meaningful NULL values (e.g., active status). Imputation blocked to prevent data corruption."
            }

        # If missingness is extremely high (> 60%), it is MNAR or structurally sparse
        if missing_pct > 60.0:
            return {
                "type": "MNAR / Sparse",
                "should_impute": False,
                "reason": f"Column '{col}' is structurally sparse ({missing_pct:.1f}% missing). Imputation is highly unreliable."
            }

        # 2. Check correlation of missingness with other features (MAR vs MCAR)
        # Create missingness indicator
        missing_indicator = df[col].isna().astype(int)
        
        numeric_df = df.select_dtypes(include=[np.number])
        max_corr = 0.0
        
        for c in numeric_df.columns:
            if c == col: continue
            try:
                corr = abs(missing_indicator.corr(df[c]))
                if pd.notna(corr) and corr > max_corr:
                    max_corr = corr
            except:
                pass

        if max_corr > 0.15:
            return {
                "type": "MAR (Missing At Random)",
                "should_impute": True,
                "reason": f"Missingness correlates with other variables (max correlation {max_corr:.2f}). Safe to impute using model predictions."
            }
        else:
            return {
                "type": "MCAR (Missing Completely At Random)",
                "should_impute": True,
                "reason": "Missingness appears random relative to numeric features. Imputation safe using simple models or baseline statistics."
            }
