import pandas as pd
import numpy as np
import io

class MetadataAnalyzer:
    def analyze_file(self, file_content: bytes, filename: str):
        try:
            from data_layer.loaders.universal_loader import DataLoaderFactory
        except ImportError:
            # Fallback for when core is run directly or from different cwd
            import sys, os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from data_layer.loaders.universal_loader import DataLoaderFactory
            
        try:
            df = DataLoaderFactory.load_data(io.BytesIO(file_content), file_name=filename)
        except Exception as e:
            raise ValueError(f"Unsupported file format or error loading file: {e}")

        # 1. Sample Cleaning
        df_sample = df.head(3).copy()
        df_sample = df_sample.replace({np.nan: None})
        for col in df_sample.columns:
            if pd.api.types.is_datetime64_any_dtype(df_sample[col]):
                df_sample[col] = df_sample[col].astype(str)
        
        sample = df_sample.to_dict(orient='records')

        metadata = {
            "file_name": filename,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns_info": [],
            "sample_data": sample,
            "correlations": [],
            "system_warnings": []
        }


        # 2. Column Analysis
        for col in df.columns:
            col_data = df[col]
            dtype = str(col_data.dtype)
            # Count both true NaN and known error/dirty tokens as "missing"
            _ERROR_TOKENS = {"ERROR", "error", "UNKNOWN", "unknown", "?", "-",
                             "Not Started", "Null", "NULL", "N/A", "n/a", "na",
                             "NA", "#VALUE!", "??", "---", "#N/A", "#REF!"}
            nan_mask = col_data.isna()
            if col_data.dtype == object:
                error_mask = col_data.astype(str).str.strip().isin(_ERROR_TOKENS)
                dirty_mask = nan_mask | error_mask
            else:
                dirty_mask = nan_mask
            missing_count = int(dirty_mask.sum())
            missing_pct = round((missing_count / len(df)) * 100, 2)
            
            # Min/Max for Numeric
            min_val, max_val = None, None
            if pd.api.types.is_numeric_dtype(col_data):
                min_val = float(col_data.min()) if not col_data.empty else None
                max_val = float(col_data.max()) if not col_data.empty else None

            # Semantic Type Inference
            semantic_type = "Unknown"
            if pd.api.types.is_numeric_dtype(col_data):
                semantic_type = "Numeric"
                if col_data.nunique() == 2: semantic_type = "Binary"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                semantic_type = "DateTime"
            elif col_data.nunique() < 20 and len(col_data) > 20:
                semantic_type = "Categorical"
            else:
                semantic_type = "Text"
            
            # --- New Context Awareness Logic ---
            col_name_lower = str(col).lower()
            
            # 1. Sensitive Column Detection
            sensitive_keywords = ['id', 'uuid', 'guid', 'email', 'name', 'phone', 'address', 'password', 'token', 'secret']
            is_sensitive = False
            
            if any(keyword in col_name_lower for keyword in sensitive_keywords):
                is_sensitive = True
            elif semantic_type in ["Text", "Numeric"] and col_data.nunique() > 0:
                # If > 95% unique, it's likely an ID/Sensitive column
                if (col_data.nunique() / len(df)) >= 0.95:
                    is_sensitive = True

            # 2. Target Column Detection
            target_keywords = ['target', 'label', 'price', 'sales', 'revenue', 'status', 'churn', 'is_', 'has_']
            is_primary_target = False
            
            if semantic_type in ["Binary", "Categorical", "Numeric"]:
                if any(keyword in col_name_lower for keyword in target_keywords):
                    is_primary_target = True

            is_target = True if semantic_type in ["Binary", "Categorical", "Numeric"] else False


            col_info = {
                "name": str(col),
                "physical_type": dtype,
                "semantic_type": semantic_type,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "unique_values": int(col_data.nunique()),
                "is_sensitive": is_sensitive,
                "is_primary_target": is_primary_target,
                "is_target_candidate": is_target,
                "min_value": min_val,
                "max_value": max_val
            }
            metadata["columns_info"].append(col_info)
            
            if is_sensitive:
                metadata["system_warnings"].append(f"Column '{col}' is marked as sensitive. Heavily replacing or deleting its values is not recommended.")
            if is_primary_target:
                metadata["system_warnings"].append(f"Column '{col}' is marked as a primary target. Accuracy is prioritized over filling all missing values.")

        # 3. Correlation Analysis (Context Awareness)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            
            # Extract upper triangle of correlation matrix without diagonal
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find highly correlated features (> 0.8)
            for i in range(len(upper_tri.columns)):
                for j in range(i+1, len(upper_tri.columns)):
                    col1 = upper_tri.columns[i]
                    col2 = upper_tri.columns[j]
                    val = upper_tri.iloc[i, j]
                    if pd.notna(val) and val >= 0.8:
                        metadata["correlations"].append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation_score": round(float(val), 2),
                            "relationship": "Strongly Correlated"
                        })
                        metadata["system_warnings"].append(f"Columns '{col1}' and '{col2}' are highly correlated (score: {round(float(val), 2)}). Actions taken on one should be consistent with the other.")

        return metadata
