import pandas as pd
import numpy as np
import io

class MetadataAnalyzer:
    def analyze_file(self, file_content: bytes = None, filename: str = "", file_path: str = None):
        import polars as pl
        import json
        import numpy as np

        # Determine source
        if file_path is not None:
            if filename.endswith('.csv'):
                lf = pl.scan_csv(file_path, infer_schema_length=10000)
            elif filename.endswith('.parquet'):
                lf = pl.scan_parquet(file_path)
            elif filename.endswith(('.xls', '.xlsx')):
                lf = pl.read_excel(file_path).lazy()
            else:
                lf = pl.read_json(file_path).lazy()
        else:
            import io
            if filename.endswith('.csv'):
                lf = pl.read_csv(io.BytesIO(file_content)).lazy()
            elif filename.endswith('.parquet'):
                lf = pl.read_parquet(io.BytesIO(file_content)).lazy()
            elif filename.endswith(('.xls', '.xlsx')):
                lf = pl.from_pandas(pd.read_excel(io.BytesIO(file_content))).lazy()
            else:
                lf = pl.from_pandas(pd.read_json(io.BytesIO(file_content))).lazy()

        # Get total row count and col count
        shape_df = lf.select([pl.len().alias("rows")]).collect()
        rows = shape_df["rows"][0]
        cols = len(lf.columns)
        
        # Get head sample (up to 3 rows)
        df_sample = lf.head(3).collect().to_pandas()
        for col in df_sample.columns:
            if pd.api.types.is_datetime64_any_dtype(df_sample[col]):
                df_sample[col] = df_sample[col].astype(str)
        sample = json.loads(df_sample.to_json(orient='records', date_format='iso'))

        metadata = {
            "file_name": filename,
            "rows": int(rows),
            "cols": int(cols),
            "columns_info": [],
            "sample_data": sample 
        }

        # Analyze columns in a single streaming pass to prevent OOM
        select_exprs = []
        for col in lf.columns:
            select_exprs.append(pl.col(col).null_count().alias(f"{col}_nulls"))
            select_exprs.append(pl.col(col).n_unique().alias(f"{col}_unique"))
            dtype = lf.schema[col]
            if dtype.is_numeric():
                select_exprs.append(pl.col(col).min().alias(f"{col}_min"))
                select_exprs.append(pl.col(col).max().alias(f"{col}_max"))

        stats_df = lf.select(select_exprs).collect(streaming=True)

        for col in lf.columns:
            dtype = lf.schema[col]
            physical_type = str(dtype)
            missing_count = int(stats_df[f"{col}_nulls"][0])
            missing_pct = round((missing_count / rows) * 100, 2) if rows > 0 else 0.0
            unique_vals = int(stats_df[f"{col}_unique"][0])

            min_val, max_val = None, None
            if dtype.is_numeric():
                val_min = stats_df[f"{col}_min"][0]
                val_max = stats_df[f"{col}_max"][0]
                if val_min is not None and np.isfinite(val_min):
                    min_val = float(val_min)
                if val_max is not None and np.isfinite(val_max):
                    max_val = float(val_max)

            # Semantic Type Inference
            semantic_type = "Unknown"
            if dtype.is_numeric():
                semantic_type = "Numeric"
                if unique_vals == 2:
                    semantic_type = "Binary"
            elif dtype.is_temporal():
                semantic_type = "DateTime"
            elif unique_vals < 20 and rows > 20:
                semantic_type = "Categorical"
            else:
                semantic_type = "Text"

            is_target = True if semantic_type in ["Binary", "Categorical", "Numeric"] else False

            col_info = {
                "name": str(col),
                "physical_type": physical_type,
                "semantic_type": semantic_type,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "unique_values": unique_vals,
                "is_target_candidate": is_target,
                "min_value": min_val,
                "max_value": max_val
            }
            metadata["columns_info"].append(col_info)
            
        return metadata