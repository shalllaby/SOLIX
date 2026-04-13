import json
import re
from groq import Groq

class StrategyManager:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def construct_prompt(self, metadata, user_goal):
        # تقليل حجم الميتاداتا لتوفير التوكنز
        slim_metadata = {
            "columns": [
                {k: v for k, v in col.items() if k in ['name', 'semantic_type', 'missing_count', 'unique_values', 'physical_type']}
                for col in metadata.get("columns_info", [])
            ],
            "rows": metadata.get("rows", 0)
        }
        
        prompt = f"""
        You are an expert Data Scientist Agent (SOL Platform V30).
        
        DATA PROFILE:
        {json.dumps(slim_metadata, indent=2)}
        
        USER GOAL:
        "{user_goal}"
        
        TASK:
        Generate exactly 3 diverse cleaning strategies:
        - "Alpha": Conservative (focus on preserving data, filling missing values safely)
        - "Beta": Balanced (standard cleaning, removing obvious outliers/duplicates)
        - "Gamma": Aggressive (focus on high accuracy, dropping suspicious rows, enforcing strict types)
        
        MAPPING RULES (Strictly map columns to these actions):
        - Numeric Columns (Salary/Price/Age) -> MUST use "remove_outliers"
        - Missing values > 0 -> "smart_impute"
        - Date columns -> "standardize_date"
        - Text with typos (City/Job) -> "fuzzy_fix"
        - Mixed numbers (e.g. '100 USD') -> "remove_outliers"
        - Emails/Phones -> "clean_pattern"
        
        OUTPUT FORMAT (STRICT JSON ARRAY OF 3 OBJECTS):
        {{
            "strategies": [
                {{
                    "strategy_name": "Alpha",
                    "model_confidence_score": 9,
                    "target_column": "ColumnName_or_None",
                    "remove_duplicates": false,
                    "cleaning_strategy": {{
                        "ColumnName": "action_key"
                    }},
                    "strategy_philosophy": "Conservative approach..."
                }},
                // ... Beta & Gamma objects here
            ]
        }}
        """
        return prompt

    def _normalize_strategy(self, raw_json, metadata):
        """
        دالة ذكية لتوحيد شكل الاستراتيجية + إجبار القواعد الصارمة (The Enforcer)
        """
        # 1. البحث عن جذر الاستراتيجية
        root = raw_json.get("ai_strategy", raw_json)
        if "cleaning_strategy" in raw_json and isinstance(raw_json["cleaning_strategy"], dict): 
            root = raw_json
        
        # 2. استخراج المعلومات الأساسية
        target_col = root.get("target_column")
        remove_dups = root.get("remove_duplicates", True)
        
        if not target_col and "cleaning_strategy" in root and isinstance(root["cleaning_strategy"], dict):
             target_col = root["cleaning_strategy"].get("target_column")

        # 3. بناء قاموس التنظيف القياسي (Flattening)
        final_plan = {}
        raw_plan = root.get("cleaning_strategy", {})
        
        # معالجة الهياكل المختلفة (Cases A, B, C, D)
        if "columns_to_clean" in raw_plan:
            for item in raw_plan["columns_to_clean"]:
                if "name" in item and "method" in item:
                    final_plan[item["name"]] = item["method"]
        
        if "outlier_detection" in raw_plan and isinstance(raw_plan["outlier_detection"], dict):
            outlier_col = raw_plan["outlier_detection"].get("column")
            if outlier_col: final_plan[outlier_col] = "remove_outliers"

        if "handling_missing_values" in root:
             for item in root["handling_missing_values"].get("columns", []):
                 if "name" in item and "imputation_method" in item:
                     if item["name"] not in final_plan:
                         final_plan[item["name"]] = item["imputation_method"]

        if isinstance(raw_plan, dict):
            for k, v in raw_plan.items():
                if isinstance(v, str) and k not in ["target_column", "imputation_method", "outlier_detection"]:
                    final_plan[k] = v

        # ---------------------------------------------------------
        # 4. 🔥 The Enforcer V2 (إجبار القواعد حتى لو الـ AI غلط)
        # ---------------------------------------------------------
        if metadata:
            for col in metadata.get("columns_info", []):
                col_name = col.get('name', '')
                dtype = col.get('physical_type', '').lower()
                semantic = col.get('semantic_type', '').lower()
                col_lower = col_name.lower()

                # A. الأرقام الصريحة (Int/Float)
                if dtype in ['int64', 'float64']:
                    is_excluded = any(x in col_lower for x in ['id', 'phone', 'mobile', 'code', 'zip', 'year'])
                    if not is_excluded:
                        if col_name not in final_plan or final_plan[col_name] == "smart_impute":
                            final_plan[col_name] = "remove_outliers"

                # B. (الجديد) الأرقام المستخبية في نصوص (Salary, Price)
                # دي اللي هتحل مشكلة الـ "USD" والـ "EGP"
                if dtype == 'object':
                    numeric_keywords = ['salary', 'price', 'amount', 'cost', 'budget', 'revenue', 'income']
                    if any(k in col_lower for k in numeric_keywords):
                        # بنجبره يعمل remove_outliers (لأنها بتشمل تنظيف النصوص في cleaner.py)
                        if col_name not in final_plan:
                            final_plan[col_name] = "remove_outliers"

                # C. تصحيح النصوص (Cities, Jobs)
                if dtype == 'object':
                    text_keywords = ['city', 'job', 'title', 'country', 'name', 'governorate']
                    if any(k in col_lower for k in text_keywords) and col_name not in final_plan:
                         final_plan[col_name] = "fuzzy_fix"

                # D. التواريخ
                if "date" in semantic or "time" in semantic or "join" in col_lower:
                    if col_name not in final_plan:
                        final_plan[col_name] = "standardize_date"

        return {
            "target_column": target_col,
            "remove_duplicates": remove_dups,
            "cleaning_strategy": final_plan,
            "reasoning": root.get("reasoning", "Strategy optimized by SOL Enforcer V2.")
        }

    def _calculate_system_confidence(self, plan_strategy, metadata):
        """
        Calculates an internal "System Confidence Score" (0-10) based on deterministic rules.
        """
        score = 10.0
        
        col_info_map = {c['name']: c for c in metadata.get('columns_info', [])}
        
        # Rule 1: Check coverage of problematic columns
        for col_name, info in col_info_map.items():
            if info.get('missing_count', 0) > 0 and col_name not in plan_strategy:
                score -= 1.0  # Penalize for missing obvious problems
        
        # Check specific actions mapped to columns
        for col_name, action in plan_strategy.items():
            info = col_info_map.get(col_name)
            if not info:
                continue
                
            semantic = info.get('semantic_type', '')
            
            # Rule 2: Sensitivity Penalty
            if info.get('is_sensitive'):
                 if action in ["drop_rows", "remove_outliers"]:
                     score -= 3.0 # Heavy penalty for risking deletion of ID/Email rows
                     
            # Rule 3: Type Mismatch
            if semantic in ["Text", "Categorical"] and action == "remove_outliers":
                score -= 0.5 # Outliyers logic usually intended for numeric
            if semantic == "DateTime" and action not in ["standardize_date", "smart_impute", "drop_rows"]:
                score -= 1.0

        return max(0.0, round(score, 1))

    def _evaluate_plans(self, plans, metadata):
        best_plan = None
        highest_score = -9999
        
        evaluated_plans_info = []

        for p in plans:
            model_confidence = p.get("model_confidence_score", 5)
            plan_strategy = p.get("cleaning_strategy", {})
            action_count = len(plan_strategy.keys())
            
            # Calculate explicit system confidence
            system_confidence = self._calculate_system_confidence(plan_strategy, metadata)
            p['system_confidence_score'] = system_confidence
            
            # Final scoring (Weighted: 40% AI, 60% System + action coverage bonus)
            score = (model_confidence * 0.4) + (system_confidence * 0.6)
            
            # Target Feature Bonus
            target_cols = [c['name'] for c in metadata.get('columns_info', []) if c.get('is_primary_target')]
            for col, action in plan_strategy.items():
                if col in target_cols and action == "smart_impute":
                    score += 1 # Bonus for targeting ML features safely
                        
            # Reward comprehensive plans slightly
            score += min(action_count * 0.2, 1) 
            
            p['internal_score'] = round(score, 2)
            evaluated_plans_info.append({
                "name": p.get("strategy_name", "Unknown"),
                "model_confidence": model_confidence,
                "system_confidence": system_confidence,
                "final_score": p['internal_score'],
                "philosophy": p.get("strategy_philosophy", "")
            })
            
            if score > highest_score:
                highest_score = score
                best_plan = p
                
        # Inject the alternative info into the winning plan so UI can read it
        if best_plan:
            best_plan['alternatives_considered'] = evaluated_plans_info
            best_plan['system_confidence_score'] = best_plan['system_confidence_score']
            best_plan['model_confidence_score'] = best_plan.get('model_confidence_score', 0)
            best_plan['reasoning'] = f"Selected {best_plan.get('strategy_name')} (Final Internal Score: {best_plan['internal_score']}). Philosophy: {best_plan.get('strategy_philosophy')}"
            
        return best_plan or plans[0]

    def generate_strategy(self, metadata, user_goal):
        prompt = self.construct_prompt(metadata, user_goal)
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON-only assistant. Output strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            
            try:
                raw_json = json.loads(response_content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    raw_json = json.loads(json_match.group())
                else:
                    raw_json = {"strategies": []} 

            plans = raw_json.get("strategies", [])
            if not plans and isinstance(raw_json, list):
                plans = raw_json
            elif not plans and isinstance(raw_json, dict):
                # Fallback if model just returns one object
                plans = [raw_json]
                
            if not plans:
                raise ValueError("No strategies returned by AI")

            winning_plan = self._evaluate_plans(plans, metadata)

            # تطبيق القواعد الصارمة على الخطة الفائزة
            normalized_strategy = self._normalize_strategy(winning_plan, metadata)
            # Retain the extra metadata we added during evaluation
            if 'alternatives_considered' in winning_plan:
                normalized_strategy['alternatives_considered'] = winning_plan['alternatives_considered']
                normalized_strategy['internal_score'] = winning_plan['internal_score']
                normalized_strategy['system_confidence_score'] = winning_plan['system_confidence_score']
                normalized_strategy['model_confidence_score'] = winning_plan['model_confidence_score']
            
            return normalized_strategy


            
        except Exception as e:
            # Fallback في حالة فشل النت
            print(f"❌ Groq Strategy Error: {str(e)}")
            fallback_strategy = self._normalize_strategy({}, metadata)
            fallback_strategy["reasoning"] = f"Fallback Strategy Active (Error: {str(e)})"
            return fallback_strategy