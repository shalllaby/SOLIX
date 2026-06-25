import json
from typing import Dict, Any, List

class LLMStrategyValidator:
    """
    Simulates Groq API response edge cases (mocking) and evaluates:
    - Hallucination Risk Score (HRS)
    - Strategy Validation Gate (Pass/Fail)
    - Routing high-risk operations to pending_review_actions.
    """
    def __init__(self):
        self.allowed_actions = {
            "drop", "remove_outliers", "smart_impute", 
            "standardize_date", "fuzzy_fix", "clean_pattern"
        }

    def generate_mock_response(self, case_type: str) -> str:
        """
        Generates simulated raw responses representing various LLM failure modes.
        """
        if case_type == "invalid_json":
            return "{'strategies': [{'strategy_name': 'Alpha', 'broken_json': true" # Unclosed JSON
            
        elif case_type == "missing_keys":
            return json.dumps({
                "strategies": [
                    {
                        "strategy_name": "Alpha",
                        "model_confidence_score": 8,
                        # cleaning_strategy is missing entirely
                        "strategy_philosophy": "Conservative"
                    }
                ]
            })
            
        elif case_type == "hallucinated_actions":
            return json.dumps({
                "strategies": [
                    {
                        "strategy_name": "Beta",
                        "model_confidence_score": 7,
                        "cleaning_strategy": {
                            "age": "truncate_column",       # Hallucinated
                            "salary": "obliterate_outliers" # Hallucinated
                        },
                        "strategy_philosophy": "Balanced",
                        "reasoning": "Standard approach."
                    }
                ]
            })
            
        elif case_type == "contradictory_instructions":
            return json.dumps({
                "strategies": [
                    {
                        "strategy_name": "Gamma",
                        "model_confidence_score": 9,
                        "cleaning_strategy": {
                            "join_date": "standardize_date",
                            "age": "remove_outliers",
                            "user_id": "drop"               # Dangerous drop on ID
                        },
                        "remove_duplicates": True,
                        "strategy_philosophy": "Aggressive",
                        "reasoning": "Attempting to drop user_id."
                    }
                ]
            })

        # Default valid response
        return json.dumps({
            "strategies": [
                {
                    "strategy_name": "Alpha",
                    "model_confidence_score": 9,
                    "cleaning_strategy": {
                        "age": "smart_impute",
                        "salary": "remove_outliers",
                        "city": "fuzzy_fix",
                        "join_date": "standardize_date"
                    },
                    "remove_duplicates": False,
                    "strategy_philosophy": "Conservative approach",
                    "reasoning": "Cleaning dirty attributes while protecting identifiers."
                }
            ]
        })

    def calculate_hallucination_risk(self, strategy: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the Hallucination Risk Score (HRS) from 0 to 100.
        Penalizes invalid actions, missing reasoning, types mismatch, and unsafe operations.
        """
        penalties = 0
        reasons = []

        root = strategy.get("ai_strategy", strategy)
        plan = root.get("cleaning_strategy", {})
        
        # 1. Check reasoning key
        reasoning = root.get("reasoning", "")
        if not reasoning or len(str(reasoning).strip()) < 10:
            penalties += 15
            reasons.append("Reasoning explanation is missing or too short.")

        # 2. Check each proposed action
        col_info_map = {c['name']: c for c in metadata.get('columns_info', [])}
        
        action_count = len(plan)
        invalid_action_count = 0
        type_mismatches = 0
        sensitive_violations = 0

        for col, action in plan.items():
            # Action validity
            if action not in self.allowed_actions:
                invalid_action_count += 1
                penalties += 20
                reasons.append(f"Hallucinated/unsupported action '{action}' on column '{col}'.")
                continue

            info = col_info_map.get(col)
            if not info:
                # Column does not exist in schema metadata
                penalties += 10
                reasons.append(f"Action mapped to non-existent column '{col}'.")
                continue

            # Data Type Mismatch checks
            physical_type = str(info.get("physical_type", "")).lower()
            if action == "remove_outliers" and "int" not in physical_type and "float" not in physical_type:
                # remove_outliers on text/object dtype
                type_mismatches += 1
                penalties += 10
                reasons.append(f"Type mismatch: 'remove_outliers' recommended on non-numeric column '{col}'.")

            # Sensitive column check
            col_lower = str(col).lower()
            is_sensitive = any(kw in col_lower for kw in ['id', 'uuid', 'guid', 'tx_', 'hash', 'secret', 'key'])
            if is_sensitive and action in ["drop", "remove_outliers", "fuzzy_fix"]:
                sensitive_violations += 1
                penalties += 30
                reasons.append(f"Safety breach: Destructive action '{action}' recommended on sensitive column '{col}'.")

        # 3. Excessive transformations check (>60% of cols modified)
        total_cols = len(metadata.get("columns_info", []))
        if total_cols > 0 and (action_count / total_cols) > 0.6:
            penalties += 15
            reasons.append(f"Excessive modification density: Proposed actions target {action_count}/{total_cols} columns.")

        hrs = max(0.0, 100.0 - penalties)
        return {
            "hallucination_risk_score": hrs,
            "penalties_incurred": penalties,
            "reasons": reasons,
            "passed_gate": hrs >= 70.0 and sensitive_violations == 0
        }

    def run_validation_gate(self, strategy: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes validation gate checks. Splits actions into:
        - approved_actions (low risk/verified)
        - auto_executed_risks (high-risk executed under Auto-Commit)
        """
        gate_status = self.calculate_hallucination_risk(strategy, metadata)
        
        root = strategy.get("ai_strategy", strategy)
        plan = root.get("cleaning_strategy", {})
        remove_dups = root.get("remove_duplicates", False)

        approved_actions = {}
        auto_executed_risks = []

        col_info_map = {c['name']: c for c in metadata.get('columns_info', [])}

        # 1. Evaluate duplicates risk
        if remove_dups:
            # If dataset has high potential row count, flag risk
            rows = metadata.get("rows", 0)
            if rows > 10000:
                auto_executed_risks.append({
                    "action": "deduplicate",
                    "column": "all",
                    "risk_level": "High",
                    "reason": f"Deduplication on large dataset ({rows} rows) can trigger unexpected row drops."
                })

        # 2. Check each column action risk profile
        for col, action in plan.items():
            info = col_info_map.get(col, {})
            missing_count = info.get("missing_count", 0)
            total_rows = metadata.get("rows", 1)
            missing_ratio = missing_count / total_rows if total_rows > 0 else 0.0

            # High Risk operations list
            is_high_risk = False
            review_reason = ""

            if action == "drop":
                is_high_risk = True
                review_reason = f"Recommends deleting column '{col}' permanently."
            elif action == "smart_impute" and missing_ratio > 0.35:
                is_high_risk = True
                review_reason = f"Imputing missing values on '{col}' where missingness ({missing_ratio*100:.1f}%) exceeds 35% threshold."
            elif action == "fuzzy_fix" and info.get("unique_values", 0) > 100:
                is_high_risk = True
                review_reason = f"Fuzzy matching text column '{col}' with high cardinality ({info.get('unique_values')} unique values) may corrupt valid keys."

            if is_high_risk:
                auto_executed_risks.append({
                    "action": action,
                    "column": col,
                    "risk_level": "High",
                    "reason": review_reason
                })
            
            approved_actions[col] = action

        return {
            "passed_gate": gate_status["passed_gate"],
            "hallucination_risk_score": gate_status["hallucination_risk_score"],
            "approved_actions": approved_actions,
            "pending_review_actions": [],  # Deprecated
            "auto_executed_risks": auto_executed_risks,
            "reasons": gate_status["reasons"]
        }
