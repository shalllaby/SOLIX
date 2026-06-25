import json
import re
from groq import Groq

class StrategyManager:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key, max_retries=0)
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
        You are an expert Data Scientist Agent (SOL Platform V29).
        
        DATA PROFILE:
        {json.dumps(slim_metadata, indent=2)}
        
        USER GOAL:
        "{user_goal}"
        
        TASK:
        Generate a precise JSON cleaning strategy.
        
        MAPPING RULES (Strictly map columns to these actions):
        - Numeric Columns (Salary/Price/Age) -> MUST use "remove_outliers"
        - Missing values > 0 -> "smart_impute"
        - Date columns -> "standardize_date"
        - Text with typos (City/Job) -> "fuzzy_fix"
        - Mixed numbers (e.g. '100 USD') -> "remove_outliers" (it handles extraction too)
        - Emails/Phones -> "clean_pattern"
        
        OUTPUT FORMAT (STRICT JSON):
        {{
            "target_column": "ColumnName_or_None",
            "remove_duplicates": true,
            "cleaning_strategy": {{
                "ColumnName": "action_key"
            }},
            "reasoning": "Brief explanation"
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

    def generate_strategy(self, metadata, user_goal):
        prompt = self.construct_prompt(metadata, user_goal)
        try:
            models = [
                self.model,
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile"
            ]
            seen = set()
            fallback_models = []
            for m in models:
                if m not in seen:
                    seen.add(m)
                    fallback_models.append(m)

            chat_completion = None
            last_err = None

            for i, model_name in enumerate(fallback_models):
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a JSON-only assistant. Output strict JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        model=model_name,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    break
                except Exception as e:
                    last_err = e
                    is_rate_limit = False
                    if hasattr(e, "status_code") and e.status_code == 429:
                        is_rate_limit = True
                    elif "RateLimit" in type(e).__name__:
                        is_rate_limit = True
                        
                    if is_rate_limit and i < len(fallback_models) - 1:
                        next_model = fallback_models[i + 1]
                        print(f"Model {model_name} hit rate limit, falling back to Model {next_model}...")
                        continue
                    else:
                        raise e

            if chat_completion is None:
                if last_err:
                    raise last_err
                else:
                    raise RuntimeError("All LLM models failed to respond.")
            
            response_content = chat_completion.choices[0].message.content
            
            try:
                raw_json = json.loads(response_content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    raw_json = json.loads(json_match.group())
                else:
                    raw_json = {} 

            # تطبيق القواعد الصارمة
            normalized_strategy = self._normalize_strategy(raw_json, metadata)
            
            return normalized_strategy
            
        except Exception as e:
            # Fallback في حالة فشل النت
            print(f"❌ Groq Strategy Error: {str(e)}")
            fallback_strategy = self._normalize_strategy({}, metadata)
            fallback_strategy["reasoning"] = f"Fallback Strategy Active (Error: {str(e)})"
            return fallback_strategy