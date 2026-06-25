import json
import logging
from typing import Dict, Any, List, Tuple

class EnterprisePolicyEngine:
    """
    Enterprise Policy Engine:
    - Enforces governance rules (protected columns, max row drops).
    - Checks risk thresholds.
    - Routes high-risk actions to human review queues.
    """
    def __init__(self, policy_config: Dict[str, Any] = None):
        # Default policy configuration
        self.policy = policy_config or {}
        if "protected_columns" not in self.policy:
            self.policy["protected_columns"] = ["id", "uuid", "guid", "tx_", "transaction", "hash", "secret", "iban", "account", "token", "key"]
        if "restricted_actions" not in self.policy:
            self.policy["restricted_actions"] = {
                "drop": ["id", "uuid", "guid", "email", "phone", "name"],
                "remove_outliers": ["id", "uuid", "guid", "email", "phone", "name"],
                "fuzzy_fix": ["id", "uuid", "guid", "email", "phone"]
            }
        if "max_row_deletion_percentage" not in self.policy:
            self.policy["max_row_deletion_percentage"] = 5.0
        if "risk_thresholds" not in self.policy:
            self.policy["risk_thresholds"] = {
                "High": "require_review",
                "Medium": "allow_with_warning",
                "Low": "auto_execute"
            }
        self.approved_actions = self.policy.get("approved_actions", [])
        self.pending_review = []
        self.blocked_actions = []
        self.warnings = []
        self.auto_executed_risks = []

    def is_protected(self, col: str) -> bool:
        col_lower = str(col).lower()
        return any(keyword in col_lower for keyword in self.policy.get("protected_columns", []))

    def validate_action(self, action: str, column: str, df_shape: Tuple[int, int], expected_impact: Dict[str, Any] = None) -> str:
        """
        Validates an action against the governance rules.
        Returns:
            "execute" - action can proceed automatically.
            "block" - action is blocked entirely due to safety policy.
        """
        col_lower = str(column).lower()
        
        # Check if pre-approved by user review override (kept for compatibility)
        for app_act in self.approved_actions:
            if app_act.get("column") == column and app_act.get("action") == action:
                self.warnings.append(f"Action '{action}' on '{column}' executed via user approval.")
                return "execute"
        
        # 1. Protected Columns validation
        if self.is_protected(column):
            restricted_for_action = self.policy.get("restricted_actions", {}).get(action, [])
            if any(keyword in col_lower for keyword in restricted_for_action) or action == "drop":
                self.blocked_actions.append(f"Blocked action '{action}' on protected column '{column}'")
                return "block"

        # 2. Risk & Impact validation
        if expected_impact:
            # Check row drop limits
            rows_removed = expected_impact.get("rows_removed", 0)
            total_rows = df_shape[0]
            if total_rows > 0:
                pct_dropped = (rows_removed / total_rows) * 100
                if pct_dropped > self.policy.get("max_row_deletion_percentage", 5.0):
                    self.auto_executed_risks.append({
                        "action": action,
                        "column": column,
                        "reason": f"Row drop percentage ({pct_dropped:.2f}%) exceeds policy limit ({self.policy.get('max_row_deletion_percentage')}%)."
                    })
                    self.warnings.append(f"Auto-executed: Row drop percentage ({pct_dropped:.2f}%) exceeded policy limit ({self.policy.get('max_row_deletion_percentage')}%).")
                    return "execute"

        # 3. Action-specific risk classification
        risk_level = self.get_action_risk(action, column)
        action_policy = self.policy.get("risk_thresholds", {}).get(risk_level, "auto_execute")
        
        if action_policy == "require_review":
            self.auto_executed_risks.append({
                "action": action,
                "column": column,
                "reason": f"Action classified as {risk_level} risk."
            })
            self.warnings.append(f"Auto-executed high-risk action '{action}' on column '{column}'.")
            return "execute"
        elif action_policy == "allow_with_warning":
            self.warnings.append(f"Medium risk action '{action}' allowed on column '{column}' with warning.")
            return "execute"

        return "execute"

    def get_action_risk(self, action: str, column: str) -> str:
        if action in ["drop", "remove_duplicates"] or self.is_protected(column):
            return "High"
        elif action in ["smart_impute", "fuzzy_fix"]:
            return "Medium"
        return "Low"
