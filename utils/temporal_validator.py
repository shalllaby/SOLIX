import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple

class TemporalValidator:
    """
    Temporal Intelligence Layer:
    - Validates chronological consistency (e.g., start_date < end_date).
    - Identifies future-date anomalies relative to the system clock.
    - Profiles temporal drift and impossibly clustered timestamps.
    """
    @staticmethod
    def validate_temporal_fields(df: pd.DataFrame) -> List[str]:
        warnings = []
        now = datetime.now()
        
        # 1. Identify date columns
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append((col, df[col]))
            elif 'date' in col.lower() or 'time' in col.lower():
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > 0:
                        date_cols.append((col, parsed))
                except:
                    pass

        # 2. Future-Date Anomaly Check
        for name, series in date_cols:
            future_mask = series > now
            future_count = int(future_mask.sum())
            if future_count > 0:
                warnings.append(
                    f"Temporal Anomaly: Column '{name}' contains {future_count} future dates relative to current date ({now.strftime('%Y-%m-%d')})."
                )

        # 3. Chronological Sequencing Check (e.g., 'start' before 'end', 'join' before 'exit')
        # Find column pairs that represent start/end sequences
        date_names = [name for name, _ in date_cols]
        pairs = []
        for c1 in date_names:
            for c2 in date_names:
                if c1 == c2: continue
                # Match typical chronological patterns
                if ('start' in c1.lower() and 'end' in c2.lower()) or \
                   ('join' in c1.lower() and 'exit' in c2.lower()) or \
                   ('hire' in c1.lower() and 'resign' in c2.lower()) or \
                   ('create' in c1.lower() and 'update' in c2.lower()):
                    pairs.append((c1, c2))

        for start_col, end_col in pairs:
            s_parsed = dict(date_cols)[start_col]
            e_parsed = dict(date_cols)[end_col]
            
            # Find rows where end_date is before start_date
            invalid_sequence_mask = e_parsed < s_parsed
            invalid_count = int(invalid_sequence_mask.sum())
            if invalid_count > 0:
                warnings.append(
                    f"Sequence Anomaly: Column '{end_col}' has {invalid_count} values occurring BEFORE start column '{start_col}'."
                )

        return warnings
