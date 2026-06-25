import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import warnings
import logging
from groq import Groq
from backend.utils.llm_logger import log_groq_response
import mimetypes
import re
import io

warnings.filterwarnings('ignore')

logger = logging.getLogger("SOL.SemanticProcessor")

class SemanticProcessor:
    """
    Core logic for AI-driven semantic mapping.
    Converts categorical/binary columns to numeric based on LLM understanding.
    """
    
    def __init__(self, model: str = "llama-3.3-70b-versatile", cache_enabled: bool = True, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.cache_enabled = cache_enabled
        self.decision_cache: Dict[str, Dict] = {}
        
        self.results = {
            'total_columns': 0,
            'binary_columns_found': 0,
            'multiclass_columns_found': 0,
            'columns_converted': 0,
            'conversion_details': {},
            'errors': [],
            'llm_decisions': {}
        }

    def _get_cache_key(self, val1: str, val2: str, col_name: str) -> str:
        key_str = f"{col_name}_{val1}_{val2}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_numeric_column(self, df: pd.DataFrame, col: str) -> bool:
        try:
            pd.to_numeric(df[col].dropna())
            return True
        except (ValueError, TypeError):
            return False

    def _detect_binary_columns(self, df: pd.DataFrame) -> List[str]:
        binary_cols = []
        for col in df.columns:
            if self._is_numeric_column(df, col):
                continue
            unique_vals = df[col].dropna().nunique()
            if unique_vals == 2:
                binary_cols.append(col)
        return binary_cols

    def _detect_multiclass_columns(self, df: pd.DataFrame) -> List[str]:
        multiclass_cols = []
        for col in df.columns:
            if self._is_numeric_column(df, col):
                continue
            unique_vals = df[col].dropna().nunique()
            if 2 < unique_vals <= 10:
                if df[col].dtype == 'object':
                    multiclass_cols.append(col)
        return multiclass_cols

    def _get_llm_decision(self, val1: str, val2: str, col_name: str, api_key: Optional[str] = None) -> Dict:
        try:
            resolved_key = api_key or self.api_key or os.getenv('GROQ_API_KEY')
            if not resolved_key:
                raise ValueError("Groq API Key is not configured.")
            client_to_use = Groq(api_key=resolved_key, max_retries=0)

            prompt = f"""Analyze these two values from column '{col_name}':
Value 1: {val1}
Value 2: {val2}

Determine which value represents a positive/affirmative/active state (should map to 1) and which represents a negative/passive/inactive state (should map to 0).

Respond ONLY with valid JSON:
{{
  "positive_value": "value that should be 1",
  "negative_value": "value that should be 0",
  "reasoning": "brief explanation in English",
  "confidence": 0.95
}}"""
            
            response = None
            last_err = None
            
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

            for i, model_name in enumerate(fallback_models):
                try:
                    response = client_to_use.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.2
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
                        logger.warning(f"Model {model_name} hit rate limit, falling back to Model {next_model}...")
                        continue
                    else:
                        raise e
            
            if response is None:
                if last_err:
                    raise last_err
                else:
                    raise RuntimeError("All LLM models failed to respond.")
            
            # Log Groq token usage
            try:
                log_groq_response(response, module_name="semantic")
            except Exception as e_log:
                logger.warning(f"Failed to log token usage: {e_log}")

            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = json.loads(response_text)
            
            return decision
        except Exception as e:
            return {
                "positive_value": val1,
                "negative_value": val2,
                "reasoning": f"Default fallback due to error: {str(e)}",
                "confidence": 0.5
            }

    def process_dataframe(self, df: pd.DataFrame, api_key: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        df_converted = df.copy()
        self.results['total_columns'] = len(df.columns)
        
        binary_cols = self._detect_binary_columns(df)
        multiclass_cols = self._detect_multiclass_columns(df)
        
        self.results['binary_columns_found'] = len(binary_cols)
        self.results['multiclass_columns_found'] = len(multiclass_cols)
        
        # 1. Process Binary Columns with AI
        for col in binary_cols:
            try:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    val1, val2 = str(unique_vals[0]), str(unique_vals[1])
                    cache_key = self._get_cache_key(val1, val2, col)
                    
                    if self.cache_enabled and cache_key in self.decision_cache:
                        decision = self.decision_cache[cache_key]
                    else:
                        decision = self._get_llm_decision(val1, val2, col, api_key=api_key)
                        if self.cache_enabled:
                            self.decision_cache[cache_key] = decision
                    
                    mapping = {
                        decision['positive_value']: 1,
                        decision['negative_value']: 0
                    }
                    
                    # Convert to numeric mapping
                    df_converted[col] = df[col].astype(str).map(mapping)
                    
                    self.results['conversion_details'][col] = {
                        'original_values': [val1, val2],
                        'mapping': mapping,
                        'reasoning': decision.get('reasoning', 'No reasoning provided'),
                        'confidence': decision.get('confidence', 0.5),
                        'type': 'binary'
                    }
                    self.results['columns_converted'] += 1
            except Exception as e:
                self.results['errors'].append(f"Error mapping column {col}: {str(e)}")

        # 2. Process Multiclass Columns with Label Encoding (Standard)
        for col in multiclass_cols:
            try:
                unique_vals = df[col].dropna().unique()
                mapping = {str(val): i for i, val in enumerate(sorted(unique_vals))}
                df_converted[col] = df[col].astype(str).map(mapping)
                
                self.results['conversion_details'][col] = {
                    'original_values': [str(v) for v in unique_vals],
                    'mapping': mapping,
                    'reasoning': "Automated multiclass label encoding",
                    'confidence': 1.0,
                    'type': 'multiclass'
                }
                self.results['columns_converted'] += 1
            except Exception as e:
                self.results['errors'].append(f"Error mapping multiclass column {col}: {str(e)}")
        
        return df_converted, self.results

def load_universal_file(content: bytes, filename: str) -> pd.DataFrame:
    """Helper to load different formats into a DataFrame."""
    ext = os.path.splitext(filename)[1].lower()
    file_obj = io.BytesIO(content)
    
    if ext == '.csv':
        return pd.read_csv(file_obj)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_obj)
    elif ext == '.json':
        return pd.read_json(file_obj)
    elif ext == '.parquet':
        return pd.read_parquet(file_obj)
    elif ext == '.tsv':
        return pd.read_csv(file_obj, sep='\t')
    else:
        # Generic fallback
        return pd.read_csv(file_obj)
