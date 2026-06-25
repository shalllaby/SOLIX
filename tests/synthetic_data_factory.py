import numpy as np
import pandas as pd
import hashlib
import uuid
from typing import Dict, Any, Tuple

class SyntheticDataFactory:
    """
    Generates realistic synthetic datasets and injects controlled corruptions
    (missing values, outliers, dirty dates, fuzzy spelling variations, currency markers,
    and Arabic/multilingual noise) to support rigorous cleaner evaluation.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def generate_ground_truth(self, n_rows: int = 1000) -> pd.DataFrame:
        """
        Generates a clean reference dataset.
        """
        np.random.seed(self.seed)
        
        user_ids = [f"USR-{10000 + i}" for i in range(n_rows)]
        tx_hashes = [hashlib.sha256(f"tx_{i}_{self.seed}".encode()).hexdigest()[:16] for i in range(n_rows)]
        ages = np.random.randint(18, 70, size=n_rows).astype(float)
        salaries = np.random.randint(3000, 25000, size=n_rows).astype(float)
        cities = np.random.choice(["Cairo", "Alexandria", "Giza", "Luxor", "Aswan"], size=n_rows)
        join_dates = pd.date_range(start="2021-01-01", periods=n_rows, freq="h").strftime("%Y-%m-%d")
        
        # Multilingual Arabic values
        feedback_options = [
            "خدمة ممتازة جدا ورائعة",
            "تجربة سيئة ولن أكررها",
            "الدعم الفني بطيء للغاية",
            "مقبول ولكن يحتاج تطوير",
            "المنتج رائع وسريع التوصيل"
        ]
        arabic_feedback = np.random.choice(feedback_options, size=n_rows)

        df = pd.DataFrame({
            "user_id": user_ids,
            "tx_hash": tx_hashes,
            "age": ages,
            "salary": salaries,
            "city": cities,
            "join_date": join_dates,
            "arabic_feedback": arabic_feedback
        })
        return df

    def inject_corruption(self, df: pd.DataFrame, 
                          missing_rate: float = 0.10, 
                          outlier_rate: float = 0.02,
                          date_corrupt_rate: float = 0.10,
                          typo_rate: float = 0.10,
                          currency_rate: float = 0.15,
                          arabic_num_rate: float = 0.10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Injects controlled corruptions into a clean DataFrame.
        Returns the corrupted DataFrame and a metadata mapping of affected indices.
        """
        np.random.seed(self.seed)
        corrupted_df = df.copy()
        n_rows = len(df)
        
        metrics_meta = {
            "age_missing": [],
            "salary_missing": [],
            "salary_outliers": [],
            "join_date_corrupted": [],
            "city_typos": [],
            "salary_currency_noise": [],
            "arabic_num_corrupted": []
        }

        # 1. Missing values
        for col in ["age", "salary"]:
            mask = np.random.rand(n_rows) < missing_rate
            corrupted_df.loc[mask, col] = np.nan
            metrics_meta[f"{col}_missing"] = np.where(mask)[0].tolist()

        # 2. Outliers (numeric values multiplied/skewed)
        outlier_mask = (np.random.rand(n_rows) < outlier_rate) & (corrupted_df["salary"].notna())
        corrupted_df.loc[outlier_mask, "salary"] = corrupted_df.loc[outlier_mask, "salary"] * 10.0
        metrics_meta["salary_outliers"] = np.where(outlier_mask)[0].tolist()

        # 3. Date format corruption (converting YYYY-MM-DD to DD/MM/YYYY or text formats)
        date_mask = np.random.rand(n_rows) < date_corrupt_rate
        formats = ["%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "invalid_date_text"]
        for idx in np.where(date_mask)[0]:
            orig = pd.to_datetime(df.loc[idx, "join_date"])
            fmt = np.random.choice(formats)
            if fmt == "invalid_date_text":
                corrupted_df.loc[idx, "join_date"] = "غير معروف"
            else:
                corrupted_df.loc[idx, "join_date"] = orig.strftime(fmt)
        metrics_meta["join_date_corrupted"] = np.where(date_mask)[0].tolist()

        # 4. Text typos & casing changes
        typo_mask = np.random.rand(n_rows) < typo_rate
        for idx in np.where(typo_mask)[0]:
            orig = df.loc[idx, "city"]
            # Change case or introduce typo
            if np.random.rand() > 0.5:
                corrupted_df.loc[idx, "city"] = orig.lower()
            else:
                # Replace character
                chars = list(orig)
                chars[len(chars)//2] = 'u' if chars[len(chars)//2] != 'u' else 'x'
                corrupted_df.loc[idx, "city"] = "".join(chars)
        metrics_meta["city_typos"] = np.where(typo_mask)[0].tolist()

        # 5. Currency markings (e.g. 5000 -> "5000 USD" or "$5000")
        currency_mask = (np.random.rand(n_rows) < currency_rate) & (corrupted_df["salary"].notna())
        for idx in np.where(currency_mask)[0]:
            val = corrupted_df.loc[idx, "salary"]
            prefix_or_suffix = np.random.choice(["prefix", "suffix"])
            if prefix_or_suffix == "prefix":
                corrupted_df.loc[idx, "salary"] = f"${int(val)}"
            else:
                corrupted_df.loc[idx, "salary"] = f"{int(val)} EGP"
        metrics_meta["salary_currency_noise"] = np.where(currency_mask)[0].tolist()

        # 6. Arabic numerals (١٢٣) and Alef Hamza typo injections
        arabic_num_mask = np.random.rand(n_rows) < arabic_num_rate
        arabic_num_map = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
        for idx in np.where(arabic_num_mask)[0]:
            orig_feedback = df.loc[idx, "arabic_feedback"]
            # Replace Alef Hamza (أ) with (ا)
            modified = orig_feedback.replace("أ", "ا")
            # Append some Arabic digits
            random_digit = str(np.random.randint(1, 100)).translate(arabic_num_map)
            corrupted_df.loc[idx, "arabic_feedback"] = f"{modified} {random_digit}"
        metrics_meta["arabic_num_corrupted"] = np.where(arabic_num_mask)[0].tolist()

        return corrupted_df, metrics_meta

    @staticmethod
    def generate_fingerprint(df: pd.DataFrame) -> str:
        """
        Generates a stable dataset fingerprint using:
        - Schema shape (columns and sorted types)
        - Row counts
        - Null counts profile
        - Entropy profile of values
        """
        # 1. Schema string
        schema_info = []
        for col in sorted(df.columns):
            schema_info.append(f"{col}:{str(df[col].dtype)}")
        schema_str = "|".join(schema_info)
        
        # 2. Null profile
        nulls_str = "|".join([f"{col}:{df[col].isna().sum()}" for col in sorted(df.columns)])
        
        # 3. Shape & Row Count
        shape_str = f"rows:{len(df)}:cols:{len(df.columns)}"
        
        # 4. Value distribution hashing
        val_hashes = []
        for col in sorted(df.columns):
            non_nulls = df[col].dropna().astype(str).tolist()
            col_hash = hashlib.md5("".join(non_nulls[:100]).encode('utf-8')).hexdigest()
            val_hashes.append(col_hash)
        val_str = "|".join(val_hashes)

        full_fingerprint_payload = f"{schema_str}#{nulls_str}#{shape_str}#{val_str}"
        return hashlib.sha256(full_fingerprint_payload.encode('utf-8')).hexdigest()
