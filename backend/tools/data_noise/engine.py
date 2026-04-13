import pandas as pd
import numpy as np
import random
import string

class DataCorruptor:
    def __init__(self, df: pd.DataFrame):
        # We work on a copy to preserve the original
        self.df = df.copy()

    def get_dataframe(self):
        return self.df

    def apply_corruption(self, column, problem_type, ratio):
        if column is not None and column not in self.df.columns and column != "All Columns (Row Level)":
            raise ValueError(f"Column '{column}' not found in dataset.")
            
        if not (0 <= ratio <= 100):
            raise ValueError("Ratio must be between 0 and 100.")
            
        fraction = ratio / 100.0
        n_rows = len(self.df)
        n_to_corrupt = int(n_rows * fraction)
        
        if n_to_corrupt == 0:
            return self.df
            
        if problem_type == "Duplications":
            self._duplicate_rows(n_to_corrupt)
            return self.df
            
        # Get random integer indices to corrupt
        indices_to_corrupt = np.random.choice(self.df.index, size=n_to_corrupt, replace=False)
        
        if problem_type == "Missing Values (NaN)":
            self._inject_missing_values(column, indices_to_corrupt)
        elif problem_type == "Outliers (Numeric)":
            self._inject_outliers(column, indices_to_corrupt)
        elif problem_type == "Typos (Text)":
            self._inject_typos(column, indices_to_corrupt)
        elif problem_type == "Formatting Issues":
            self._inject_format_issues(column, indices_to_corrupt)
        elif problem_type == "Type Inconsistency":
            self._inject_type_inconsistency(column, indices_to_corrupt)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
            
        return self.df

    def _inject_missing_values(self, column, indices):
        self.df.loc[indices, column] = np.nan

    def _inject_outliers(self, column, indices):
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Outliers can only be applied to numeric columns. Column '{column}' was chosen.")
            
        for idx in indices:
            val = self.df.at[idx, column]
            if pd.isna(val):
                continue
            factor = random.uniform(5, 500)
            if random.choice([True, False]):
                self.df.at[idx, column] = val * factor
            else:
                self.df.at[idx, column] = val / factor if val != 0 else factor

    def _inject_typos(self, column, indices):
        for idx in indices:
            val = self.df.at[idx, column]
            if pd.isna(val):
                continue
            val_str = str(val)
            if len(val_str) < 2:
                continue
                
            typo_type = random.choice(['swap', 'drop', 'add'])
            pos = random.randint(0, len(val_str) - 2) if len(val_str) > 1 else 0
            
            if typo_type == 'swap' and len(val_str) > 1:
                lst = list(val_str)
                lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
                new_val = "".join(lst)
            elif typo_type == 'drop' and len(val_str) > 1:
                new_val = val_str[:pos] + val_str[pos+1:]
            elif typo_type == 'add':
                char = random.choice(string.ascii_letters)
                new_val = val_str[:pos] + char + val_str[pos:]
            else:
                new_val = val_str
                
            self.df.at[idx, column] = new_val

    def _inject_format_issues(self, column, indices):
        for idx in indices:
            val = self.df.at[idx, column]
            if pd.isna(val):
                continue
            val_str = str(val)
            issue_type = random.choice(['leading', 'trailing', 'middle'])
            if issue_type == 'leading':
                new_val = "   " + val_str
            elif issue_type == 'trailing':
                new_val = val_str + "   "
            else:
                mid = len(val_str) // 2
                new_val = val_str[:mid] + "  " + val_str[mid:]
            self.df.at[idx, column] = new_val

    def _inject_type_inconsistency(self, column, indices):
        if self.df[column].dtype != object:
            self.df[column] = self.df[column].astype(object)
        for idx in indices:
            inconsistent_val = random.choice(["Unknown", "N/A", "Error", "???", "Invalid"])
            self.df.at[idx, column] = inconsistent_val

    def _duplicate_rows(self, n_to_corrupt):
        if n_to_corrupt == 0 or len(self.df) == 0:
            return
        rows_to_duplicate = self.df.sample(n=n_to_corrupt, replace=True)
        self.df = pd.concat([self.df, rows_to_duplicate], ignore_index=True)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
