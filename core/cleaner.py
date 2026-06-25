import pandas as pd
import numpy as np
from utils.date_parser import DateParser
from utils.numeric_cleaner import NumericCleaner
from utils.ai_imputer import AIImputer
from utils.text_normalizer import TextNormalizer
from utils.pattern_cleaner import PatternCleaner
from utils.policy_engine import EnterprisePolicyEngine

class SmartDataCleaner:
    def __init__(self, df: pd.DataFrame, policy_config: dict = None):
        self.df = df.copy()
        self.report = {"actions": [], "engine_version": "V25.0_Legendary"}
        self.policy_engine = EnterprisePolicyEngine(policy_config)

    def execute_strategy(self, strategy_json: dict):
        root = strategy_json.get("ai_strategy", strategy_json)
        plan = root.get("cleaning_strategy", {})
        should_remove_duplicates = root.get("remove_duplicates", False)

        # Pre-validate all actions in the plan to avoid duplicate policy logging
        decisions = {}
        for col, action in plan.items():
            decisions[col] = self.policy_engine.validate_action(action, col, self.df.shape)

        # Validate duplicate removal action
        if should_remove_duplicates:
            dedup_decision = self.policy_engine.validate_action("remove_duplicates", "all", self.df.shape)
        else:
            dedup_decision = "execute"

        # 1. First Tap: Dedupe
        if should_remove_duplicates and dedup_decision == "execute":
            self._remove_duplicates_logic("Initial")

        # 2. Pre-cleaning & Logical Validation 🛡️
        numeric_actions = ["smart_impute", "remove_outliers", "impute_mean", "impute_mode"]
        for col, action in plan.items():
            if col in self.df.columns and action in numeric_actions:
                decision = decisions.get(col, "execute")
                if decision == "block":
                    continue

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
            
            decision = decisions.get(col, "execute")
            if decision == "block":
                continue

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
                self.df, imp_status = imputer.predict_missing_values(col, plan)
                if imp_status == "ai_predictive":
                    self.report["actions"].append(f"Smart Impute: {col}")
                elif imp_status == "fallback":
                    self.report["actions"].append(f"Final Polish: Filled remaining NaNs in '{col}'")

            elif action == "smart_impute":
                imputer = AIImputer(self.df)
                self.df, imp_status = imputer.predict_missing_values(col, plan)
                if imp_status == "ai_predictive":
                    self.report["actions"].append(f"Smart Impute: {col}")
                elif imp_status == "fallback":
                    self.report["actions"].append(f"Final Polish: Filled remaining NaNs in '{col}'")

        # 4. Pattern Recognition
        for col in self.df.columns:
            col_lower = col.lower()
            if 'email' in col_lower or 'mail' in col_lower:
                if pd.api.types.is_object_dtype(self.df[col]):
                    self.df[col] = PatternCleaner.clean_email_column(self.df[col])
            elif 'phone' in col_lower or 'mobile' in col_lower:
                self.df[col] = PatternCleaner.clean_phone_column(self.df[col])

        # 5. Second Tap: Dedupe
        if should_remove_duplicates and dedup_decision == "execute":
            self._remove_duplicates_logic("Post-Clean")

        # 6. Final Polish
        self._final_polish(plan, decisions)

        # 7. Append Policy Engine results to the report
        self.report["blocked_actions"] = self.policy_engine.blocked_actions
        self.report["pending_review_actions"] = []  # Deprecated review queue
        self.report["auto_executed_risks"] = getattr(self.policy_engine, "auto_executed_risks", [])
        self.report["warnings"] = self.policy_engine.warnings

        # Calculate truth confidence score
        blocked_count = len(self.policy_engine.blocked_actions)
        auto_risk_count = len(self.report["auto_executed_risks"])
        warning_count = len(self.policy_engine.warnings)
        
        confidence = 100.0 - (blocked_count * 25 + auto_risk_count * 15 + warning_count * 5)
        self.report["truth_confidence_score"] = max(10.0, min(100.0, confidence))

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

    def _final_polish(self, plan, decisions):
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                if plan.get(col) != "drop":
                    decision = decisions.get(col, "execute")
                    if decision == "block":
                        continue
                    imputer = AIImputer(self.df)
                    self.df, imp_status = imputer.predict_missing_values(col, plan)
                    if imp_status == "ai_predictive":
                        self.report["actions"].append(f"Smart Impute: {col}")
                    elif imp_status == "fallback":
                        self.report["actions"].append(f"Final Polish: Filled remaining NaNs in '{col}'")