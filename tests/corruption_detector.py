import pandas as pd
import numpy as np
from typing import Dict, Any, List

class CorruptionDetector:
    """
    Computes global validation metrics:
    - Exact Match Accuracy (EMA)
    - Over-cleaning Rate (OCR)
    - Corruption Rate (CR)
    - Recovery Precision metrics grouped by corruption type.
    """
    def __init__(self, ground_truth: pd.DataFrame, corrupted: pd.DataFrame, cleaned: pd.DataFrame, metadata: Dict[str, Any]):
        self.ground_truth = ground_truth
        self.corrupted = corrupted
        self.cleaned = cleaned
        self.meta = metadata

    def _values_are_equivalent(self, val1, val2, col: str = "") -> bool:
        """
        Determines semantic equivalence between ground truth/expected and actual cleaned values.
        Supports numeric tolerance, date parsing matching, and Arabic orthographic normalization.
        """
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False

        # 1. Numeric check (within 15% tolerance)
        try:
            f1 = float(val1)
            f2 = float(val2)
            if max(abs(f1), abs(f2)) == 0:
                return True
            return abs(f1 - f2) / max(abs(f1), 1.0) <= 0.15
        except:
            pass

        # 2. Date check (matching date component)
        try:
            d1 = pd.to_datetime(val1)
            d2 = pd.to_datetime(val2)
            return d1.date() == d2.date()
        except:
            pass

        # 3. String & Arabic normalization comparison
        s1 = str(val1).strip().lower()
        s2 = str(val2).strip().lower()

        # Helper to normalize Arabic orthography
        def norm_ar(s: str) -> str:
            s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            s = s.replace("ة", "ه")
            # Convert Arabic numerals
            arabic_digits = "٠١٢٣٤٥٦٧٨٩"
            western_digits = "0123456789"
            trans = str.maketrans(arabic_digits, western_digits)
            s = s.translate(trans)
            # Remove digits to match clean ground truth if digits were added as noise
            s = "".join([c for c in s if not c.isdigit()])
            return " ".join(s.split())

        # If feedback column or has Arabic characters
        if col == "arabic_feedback" or any(c in "ابتثجحخدرزسشصضطظعغفقكلمنهوي" for c in s1):
            return norm_ar(s1) == norm_ar(s2)

        return s1 == s2

    def calculate_global_metrics(self) -> Dict[str, float]:
        """
        Calculates EMA, OCR, and CR.
        """
        # Find intersecting index and columns
        common_idx = self.ground_truth.index.intersection(self.cleaned.index)
        common_cols = [c for c in self.ground_truth.columns if c in self.cleaned.columns]
        
        if len(common_idx) == 0 or not common_cols:
            return {"ema": 0.0, "ocr": 0.0, "cr": 0.0}

        total_corrupted_cells = 0
        restored_cells = 0
        
        total_safe_cells = 0
        over_cleaned_cells = 0
        
        total_cells = len(common_idx) * len(common_cols)
        invalid_cells = 0

        # Pre-cache datatypes and string representations
        for col in common_cols:
            gt_series = self.ground_truth.loc[common_idx, col]
            raw_series = self.corrupted.loc[common_idx, col]
            cleaned_series = self.cleaned.loc[common_idx, col]

            # Vectorized metrics where possible, or cell-by-cell for heterogeneous comparisons
            for idx in common_idx:
                gt_val = gt_series.loc[idx]
                raw_val = raw_series.loc[idx]
                cl_val = cleaned_series.loc[idx]

                # Determine if raw value was dirty
                raw_is_null = pd.isna(raw_val)
                gt_is_null = pd.isna(gt_val)
                cl_is_null = pd.isna(cl_val)

                # Check if cell was corrupted
                is_corrupted = False
                if raw_is_null != gt_is_null:
                    is_corrupted = True
                elif not raw_is_null and not gt_is_null:
                    # check if raw was not equivalent to ground truth
                    is_corrupted = not self._values_are_equivalent(raw_val, gt_val, col)

                if is_corrupted:
                    total_corrupted_cells += 1
                    # Check if restored correctly
                    if self._values_are_equivalent(cl_val, gt_val, col):
                        restored_cells += 1
                else:
                    total_safe_cells += 1
                    # Check if cleaner modified a correct cell
                    if not self._values_are_equivalent(cl_val, raw_val, col):
                        over_cleaned_cells += 1

                # Check for un-parsable or invalid outputs
                if cl_is_null and not gt_is_null:
                    invalid_cells += 1
                elif not cl_is_null:
                    if str(cl_val) in ['NaT', 'nan', 'NaN', 'ERROR', 'error']:
                        invalid_cells += 1

        ema = (restored_cells / total_corrupted_cells * 100) if total_corrupted_cells > 0 else 100.0
        ocr = (over_cleaned_cells / total_safe_cells * 100) if total_safe_cells > 0 else 0.0
        cr = (invalid_cells / total_cells * 100) if total_cells > 0 else 0.0

        return {
            "exact_match_accuracy": float(ema),
            "over_cleaning_rate": float(ocr),
            "corruption_rate": float(cr)
        }

    def calculate_recovery_precision(self) -> Dict[str, float]:
        """
        Segment recovery precision by corruption category.
        """
        precision_report = {}
        common_idx = self.ground_truth.index.intersection(self.cleaned.index)
        
        # 1. Missing Numeric Imputation (Age & Salary)
        missing_num_idx = list(set(self.meta.get("age_missing", []) + self.meta.get("salary_missing", [])))
        precision_report["missing_numeric"] = self._subset_match_rate(missing_num_idx, ["age", "salary"], common_idx)

        # 2. Outlier Replacements (Salary)
        outlier_idx = self.meta.get("salary_outliers", [])
        precision_report["outlier_replacement"] = self._subset_match_rate(outlier_idx, ["salary"], common_idx)

        # 3. Broken Dates Standardization
        date_idx = self.meta.get("join_date_corrupted", [])
        precision_report["broken_dates"] = self._subset_match_rate(date_idx, ["join_date"], common_idx)

        # 4. Currency Noise Cleaning
        curr_idx = self.meta.get("salary_currency_noise", [])
        precision_report["currency_noise"] = self._subset_match_rate(curr_idx, ["salary"], common_idx)

        # 5. Arabic Typos / Numerals
        ar_idx = self.meta.get("arabic_num_corrupted", [])
        precision_report["arabic_typos"] = self._subset_match_rate(ar_idx, ["arabic_feedback"], common_idx)

        return precision_report

    def _subset_match_rate(self, indices: List[int], columns: List[str], common_idx) -> float:
        valid_indices = [idx for idx in indices if idx in common_idx]
        if not valid_indices or not columns:
            return 100.0

        matches = 0
        total = 0
        for col in columns:
            for idx in valid_indices:
                gt_val = self.ground_truth.loc[idx, col]
                cl_val = self.cleaned.loc[idx, col]
                
                total += 1
                if self._values_are_equivalent(gt_val, cl_val, col):
                    matches += 1
                                
        return float(matches / total * 100) if total > 0 else 100.0
