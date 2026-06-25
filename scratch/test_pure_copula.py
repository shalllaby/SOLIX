import numpy as np
import pandas as pd
from scipy.stats import norm

class PureGaussianCopula:
    def __init__(self):
        self.mappings = {}
        self.columns = []
        self.cov_matrix = None
        self.means = None
        self.numerical_cols = []
        self.categorical_cols = []
        
    def fit(self, df: pd.DataFrame, profile: dict):
        self.columns = list(df.columns)
        n_rows = len(df)
        
        Z = np.zeros((n_rows, len(self.columns)))
        
        for i, col in enumerate(self.columns):
            col_info = profile.get(col, {})
            col_type = col_info.get("type", "numerical")
            col_data = df[col]
            
            if col_data.isnull().any():
                if col_type == "numerical":
                    fill_val = col_data.median() if not pd.isna(col_data.median()) else 0
                else:
                    fill_val = col_data.mode()[0] if not col_data.mode().empty else "missing"
                col_data_clean = col_data.fillna(fill_val)
            else:
                col_data_clean = col_data
                
            if col_type == "numerical":
                self.numerical_cols.append(col)
                self.mappings[col] = {
                    "type": "numerical",
                    "values": np.sort(col_data_clean.values)
                }
                ranks = pd.Series(col_data_clean).rank(method='average')
                u = (ranks - 0.5) / n_rows
                u = np.clip(u, 1e-6, 1 - 1e-6)
                Z[:, i] = norm.ppf(u)
            else:
                self.categorical_cols.append(col)
                cats, counts = np.unique(col_data_clean.values, return_counts=True)
                sort_idx = np.argsort(cats)
                cats = cats[sort_idx]
                
                self.mappings[col] = {
                    "type": "categorical",
                    "categories": cats,
                    "values": np.sort(col_data_clean.map({cat: idx for idx, cat in enumerate(cats)}).values)
                }
                
                encoded = col_data_clean.map({cat: idx for idx, cat in enumerate(cats)})
                ranks = pd.Series(encoded).rank(method='average')
                u = (ranks - 0.5) / n_rows
                u = np.clip(u, 1e-6, 1 - 1e-6)
                Z[:, i] = norm.ppf(u)
                
        self.means = np.mean(Z, axis=0)
        self.cov_matrix = np.cov(Z.T) + np.eye(len(self.columns)) * 1e-6
        
    def sample(self, num_rows: int) -> pd.DataFrame:
        if len(self.columns) == 1:
            z_sampled = np.random.normal(self.means[0], np.sqrt(self.cov_matrix), size=(num_rows, 1))
        else:
            z_sampled = np.random.multivariate_normal(self.means, self.cov_matrix, size=num_rows)
            
        u_sampled = norm.cdf(z_sampled)
        u_sampled = np.clip(u_sampled, 1e-6, 1 - 1e-6)
        
        synthetic_dict = {}
        for i, col in enumerate(self.columns):
            mapping = self.mappings[col]
            u_col = u_sampled[:, i]
            
            if mapping["type"] == "numerical":
                orig_vals = mapping["values"]
                idx_continuous = u_col * (len(orig_vals) - 1)
                idx_low = np.floor(idx_continuous).astype(int)
                idx_high = np.ceil(idx_continuous).astype(int)
                weight = idx_continuous - idx_low
                synth_vals = (1 - weight) * orig_vals[idx_low] + weight * orig_vals[idx_high]
                synthetic_dict[col] = synth_vals
            else:
                orig_encoded = mapping["values"]
                idx_continuous = u_col * (len(orig_encoded) - 1)
                idx_closest = np.round(idx_continuous).astype(int)
                synth_encoded = orig_encoded[idx_closest]
                
                cats = mapping["categories"]
                synth_encoded_clipped = np.clip(synth_encoded, 0, len(cats) - 1).astype(int)
                synthetic_dict[col] = [cats[val] for val in synth_encoded_clipped]
                
        return pd.DataFrame(synthetic_dict)

# Quick verification test
if __name__ == "__main__":
    df = pd.DataFrame({
        "age": [23, 45, 12, 67, 34, 56, 28, 41, 19, 52],
        "salary": [25000, 48000, 10000, 80000, 39000, 62000, 31000, 45000, 20000, 58000],
        "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"]
    })
    profile = {
        "age": {"type": "numerical"},
        "salary": {"type": "numerical"},
        "gender": {"type": "categorical"}
    }
    
    copula = PureGaussianCopula()
    copula.fit(df, profile)
    synth = copula.sample(5)
    print("Original DataFrame:\n", df)
    print("\nGenerated Synthetic DataFrame:\n", synth)
