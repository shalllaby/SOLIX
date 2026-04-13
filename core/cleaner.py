import pandas as pd
import numpy as np
from utils.date_parser import DateParser
from utils.numeric_cleaner import NumericCleaner
from utils.ai_imputer import AIImputer
from utils.text_normalizer import TextNormalizer
from utils.pattern_cleaner import PatternCleaner

class SmartDataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {"actions": [], "engine_version": "V25.0_Legendary"}

    def execute_strategy(self, strategy_json: dict):
        # ... (نفس البداية) ...
        root = strategy_json.get("ai_strategy", strategy_json)
        plan = root.get("cleaning_strategy", {})
        should_remove_duplicates = root.get("remove_duplicates", False)

        # 1. First Tap: Dedupe
        if should_remove_duplicates:
            self._remove_duplicates_logic("Initial")

        # 2. Pre-cleaning & Logical Validation (الجديد هنا) 🛡️
        numeric_actions = ["smart_impute", "remove_outliers", "impute_mean", "impute_mode"]
        for col, action in plan.items():
            if col in self.df.columns and action in numeric_actions:
                # أ) تنظيف النصوص من الأرقام
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    cleaned_col = NumericCleaner.clean_dirty_strings(self.df[col])
                    if cleaned_col.notna().sum() > 0:
                        self.df[col] = cleaned_col
                
                # ب) تصحيح القيم السالبة (Logical Bounds)
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = NumericCleaner.enforce_logical_bounds(self.df[col], col)

        dirty_tokens = ['ERROR', 'error', 'UNKNOWN', 'unknown', '?', '-', 'Not Started', 'Null']
        self.df.replace(dirty_tokens, np.nan, inplace=True)
        
        # 3. Main Cleaning Loop
        for col, action in plan.items():
            if col not in self.df.columns: continue
            
            if action == "drop":
                self.df.drop(columns=[col], inplace=True, errors='ignore')

            elif action == "standardize_date":
                self.df[col] = DateParser.standardize_column(self.df[col])
                self.report["actions"].append(f"Standardized Date: {col}")

            elif action == "fuzzy_fix":
                self.df[col] = TextNormalizer.fuzzy_fix_column(self.df[col])
                self.report["actions"].append(f"Fuzzy Matched text in: {col}")

            elif action == "remove_outliers":
                self.df, count = NumericCleaner.remove_outliers_zscore(self.df, col)
                if count > 0:
                    self.report["actions"].append(f"Removed {count} Outliers from {col}")
                imputer = AIImputer(self.df)
                self.df = imputer.predict_missing_values(col, plan)

            elif action == "smart_impute":
                imputer = AIImputer(self.df)
                self.df = imputer.predict_missing_values(col, plan)
                self.report["actions"].append(f"Smart Impute: {col}")

        # 4. Pattern Recognition
        for col in self.df.columns:
            col_lower = col.lower()
            if 'email' in col_lower or 'mail' in col_lower:
                if pd.api.types.is_object_dtype(self.df[col]):
                    self.df[col] = PatternCleaner.clean_email_column(self.df[col])
            elif 'phone' in col_lower or 'mobile' in col_lower:
                self.df[col] = PatternCleaner.clean_phone_column(self.df[col])

        # 5. Second Tap: Dedupe
        if should_remove_duplicates:
            self._remove_duplicates_logic("Post-Clean")

        # 6. Final Polish
        self._final_polish(plan)

        return self.df, self.report

    def _remove_duplicates_logic(self, stage_name):
        initial_count = len(self.df)
        subset_cols = [c for c in self.df.columns if c.lower() not in ['id', 'name', 'identifier', 'index']]
        if subset_cols:
            self.df.drop_duplicates(subset=subset_cols, inplace=True)
        else:
            self.df.drop_duplicates(inplace=True)
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.report["actions"].append(f"Removed {removed_count} duplicates ({stage_name})")

    def _final_polish(self, plan):
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                if plan.get(col) != "drop":
                    imputer = AIImputer(self.df)
                    self.df = imputer.predict_missing_values(col, plan)
                    self.report["actions"].append(f"Final Polish: Filled remaining NaNs in '{col}'")