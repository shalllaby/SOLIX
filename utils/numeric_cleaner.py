import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Any

class NumericCleaner:
    @staticmethod
    def parse_numeric_value(val: Any, agg_strategy: str = "midpoint") -> float:
        """
        Context-aware numeric value parser.
        Converts percentages, currencies, compact suffixes, accounting negatives,
        scientific notation, and ranges into float while protecting semantic text.
        """
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        
        val_str = str(val).strip()
        if not val_str:
            return np.nan

        # 1. Protect Semantic Text (e.g. "Room 404", "ID-2039")
        # Remove expected symbols, digits, currency codes, ranges keywords
        allowed_vocab = r'[\d.,\s$€£¥%-]|(?:usd|eur|gbp|egp|to|next|month|q[1-4])'
        clean_text_check = re.sub(allowed_vocab, '', val_str, flags=re.IGNORECASE)
        if len(clean_text_check) > 0:
            # Check if it is a single valid compact suffix character
            if not (len(clean_text_check) == 1 and clean_text_check[0].upper() in ['K', 'M', 'B', 'E']):
                return np.nan

        # 2. Scientific Notation Check (e.g., "1.2e-4" or "3.5E6")
        sci_match = re.match(r'^\s*([-+]?[\d.]+[eE][-+]?\d+)\s*$', val_str)
        if sci_match:
            try:
                return float(sci_match.group(1))
            except:
                pass

        # 3. Accounting Negatives Check (e.g. "(500)", "($1,200.50)")
        accounting_match = re.match(r'^\s*\(\s*([^)]+)\s*\)\s*$', val_str)
        if accounting_match:
            inner_val = accounting_match.group(1)
            parsed_inner = NumericCleaner.parse_numeric_value(inner_val, agg_strategy)
            if not pd.isna(parsed_inner):
                return -1.0 * abs(parsed_inner)

        # 4. Numeric Ranges Check (e.g., "10-20 USD", "1.5K - 3K")
        range_match = re.search(
            r'^\s*([$€£¥\s]*[\d,.]+[kKmMbB%]?)\s*(?:-|to)\s*([$€£¥\s]*[\d,.]+[kKmMbB%]?)\s*(.*)$',
            val_str,
            re.IGNORECASE
        )
        if range_match:
            raw_min = range_match.group(1)
            raw_max = range_match.group(2)
            unit = range_match.group(3).strip()
            
            min_val = NumericCleaner.parse_numeric_value(raw_min, agg_strategy)
            max_val = NumericCleaner.parse_numeric_value(raw_max, agg_strategy)
            
            if not pd.isna(min_val) and not pd.isna(max_val):
                if agg_strategy == "min":
                    return min_val
                elif agg_strategy == "max":
                    return max_val
                elif agg_strategy == "weighted":
                    return 0.6 * min_val + 0.4 * max_val
                else: # midpoint
                    return (min_val + max_val) / 2.0

        # 5. Percentages
        is_pct = '%' in val_str

        # 6. Compact Suffixes (K, M, B)
        multiplier = 1.0
        suffix_match = re.search(r'([\d,.]+)\s*([kKmMbB])(?!\w)', val_str)
        if suffix_match:
            suffix = suffix_match.group(2).upper()
            if suffix == 'K':
                multiplier = 1000.0
            elif suffix == 'M':
                multiplier = 1000000.0
            elif suffix == 'B':
                multiplier = 1000000000.0
            val_str = val_str.replace(suffix_match.group(0), suffix_match.group(1))

        # 7. Extract digits, decimal point, and sign
        sign = 1.0
        if val_str.lstrip().startswith('-'):
            sign = -1.0
            val_str = val_str.lstrip()[1:]

        # Strip everything except digits and first decimal point
        digits_only = re.sub(r'[^\d.]', '', val_str)
        if digits_only.count('.') > 1:
            parts = digits_only.split('.')
            digits_only = parts[0] + '.' + ''.join(parts[1:])

        try:
            parsed_num = float(digits_only) * multiplier * sign
            if is_pct:
                parsed_num /= 100.0
            return parsed_num
        except:
            return np.nan

    @staticmethod
    def clean_dirty_strings(series: pd.Series, agg_strategy: str = "midpoint") -> pd.Series:
        """
        Applies parse_numeric_value to a pandas Series.
        """
        return series.apply(lambda x: NumericCleaner.parse_numeric_value(x, agg_strategy))

    @staticmethod
    def remove_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> Tuple[pd.DataFrame, int]:
        """
        Hybrid Outlier Engine selecting between MAD, IQR, or Percentiles based on distribution properties.
        """
        if col not in df.columns:
            return df, 0
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            return df, 0

        skew = col_data.skew()
        n = len(col_data)
        
        outliers_mask = pd.Series(False, index=df.index)

        if n < 15:
            q_low = col_data.quantile(0.01)
            q_high = col_data.quantile(0.99)
            outliers_mask = (df[col] < q_low) | (df[col] > q_high)
        elif abs(skew) > 1.5:
            q25 = col_data.quantile(0.25)
            q75 = col_data.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        else:
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            if mad > 0:
                mad_z_score = 0.6745 * (df[col] - median) / mad
                outliers_mask = np.abs(mad_z_score) > threshold
            else:
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    outliers_mask = np.abs((df[col] - mean) / std) > threshold

        outliers_count = int(outliers_mask.sum())
        # Immutably return modified copy
        df_copy = df.copy()
        df_copy.loc[outliers_mask, col] = np.nan
        return df_copy, outliers_count

    @staticmethod
    def enforce_logical_bounds(series: pd.Series, col_name: str) -> pd.Series:
        """
        Clips strictly non-negative columns without sign flipping.
        """
        col_lower = col_name.lower()
        positive_keywords = ['age', 'experience', 'quantity', 'count']
        
        if any(keyword in col_lower for keyword in positive_keywords):
            return series.clip(lower=0.0)
            
        return series