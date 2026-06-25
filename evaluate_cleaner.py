import pandas as pd
import numpy as np
import os
import sys

# Add current path to sys.path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cleaner import SmartDataCleaner

def generate_synthetic_data(n_rows=200):
    """Generate clean base dataset as Ground Truth"""
    np.random.seed(42)
    data = {
        "id": range(1, n_rows + 1),
        "age": np.random.randint(20, 60, size=n_rows).astype(float),
        "salary": np.random.randint(3000, 15000, size=n_rows).astype(float),
        "city": np.random.choice(["Cairo", "Alexandria", "Giza", "Aswan"], size=n_rows),
        "join_date": pd.date_range(start="2020-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d")
    }
    return pd.DataFrame(data)

def corrupt_data(df, corruption_rate=0.15):
    """Corrupt data by injecting NaNs, format issues, and casing changes"""
    df_corrupted = df.copy()
    n_rows = len(df)
    
    # 1. Hide numeric values
    mask_age = np.random.rand(n_rows) < corruption_rate
    mask_sal = np.random.rand(n_rows) < corruption_rate
    df_corrupted.loc[mask_age, "age"] = np.nan
    df_corrupted.loc[mask_sal, "salary"] = np.nan
    
    # 2. Corrupt date formats (change formatting style, keep value)
    mask_date = np.random.rand(n_rows) < corruption_rate
    for idx in np.where(mask_date)[0]:
        orig_date = pd.to_datetime(df.loc[idx, "join_date"])
        # Format as DD/MM/YYYY
        df_corrupted.loc[idx, "join_date"] = orig_date.strftime("%d/%m/%Y")
    
    # 3. Change casing for text values
    mask_city = np.random.rand(n_rows) < corruption_rate
    df_corrupted.loc[mask_city, "city"] = df_corrupted.loc[mask_city, "city"].str.lower()
    
    return df_corrupted, {
        "corrupted_age_indices": np.where(mask_age)[0],
        "corrupted_salary_indices": np.where(mask_sal)[0],
        "corrupted_date_indices": np.where(mask_date)[0],
        "corrupted_city_indices": np.where(mask_city)[0]
    }

def calculate_accuracy():
    print("==================================================================")
    print("                EVALUATING CLEANER CORE ACCURACY                  ")
    print("==================================================================")
    
    # 1. Setup and Corrupt Data
    df_clean = generate_synthetic_data()
    df_corrupted, targets = corrupt_data(df_clean)
    
    print(f"Dataset Size: {len(df_clean)} rows")
    print(f"Corrupted cells to reconstruct: ")
    print(f" - Missing Ages: {len(targets['corrupted_age_indices'])}")
    print(f" - Missing Salaries: {len(targets['corrupted_salary_indices'])}")
    print(f" - Non-standard Dates: {len(targets['corrupted_date_indices'])}")
    print(f" - Lowercase Cities: {len(targets['corrupted_city_indices'])}")
    print("------------------------------------------------------------------")
    
    # 2. Build cleaning strategy
    strategy = {
        "remove_duplicates": False,
        "cleaning_strategy": {
            "age": "smart_impute",
            "salary": "smart_impute",
            "city": "fuzzy_fix",
            "join_date": "standardize_date"
        }
    }
    
    # 3. Execute cleaner
    cleaner = SmartDataCleaner(df_corrupted)
    df_cleaned, report = cleaner.execute_strategy(strategy)
    
    # 4. Calculate accuracy
    
    # Age Imputation
    age_idx = targets['corrupted_age_indices']
    original_ages = df_clean.loc[age_idx, "age"]
    imputed_ages = df_cleaned.loc[age_idx, "age"]
    age_errors = np.abs(original_ages - imputed_ages)
    avg_age_error = np.mean(age_errors) if len(age_idx) > 0 else 0
    mean_age = df_clean["age"].mean()
    age_accuracy = max(0.0, 100.0 - (avg_age_error / mean_age * 100))
    
    # Salary Imputation
    sal_idx = targets['corrupted_salary_indices']
    original_sal = df_clean.loc[sal_idx, "salary"]
    imputed_sal = df_cleaned.loc[sal_idx, "salary"]
    sal_errors = np.abs(original_sal - imputed_sal)
    avg_sal_error = np.mean(sal_errors) if len(sal_idx) > 0 else 0
    mean_sal = df_clean["salary"].mean()
    sal_accuracy = max(0.0, 100.0 - (avg_sal_error / mean_sal * 100))
    
    # Date Standardization
    date_idx = targets['corrupted_date_indices']
    # Format both series to YYYY-MM-DD strings for comparison
    orig_dates = pd.to_datetime(df_clean.loc[date_idx, "join_date"]).dt.strftime("%Y-%m-%d")
    cleaned_dates = pd.to_datetime(df_cleaned.loc[date_idx, "join_date"], errors='coerce').dt.strftime("%Y-%m-%d")
    correct_dates = (orig_dates == cleaned_dates).sum()
    date_accuracy = (correct_dates / len(date_idx) * 100) if len(date_idx) > 0 else 100.0
    
    # Fuzzy Normalization
    city_idx = targets['corrupted_city_indices']
    orig_cities = df_clean.loc[city_idx, "city"]
    cleaned_cities = df_cleaned.loc[city_idx, "city"]
    correct_cities = (orig_cities == cleaned_cities).sum()
    city_accuracy = (correct_cities / len(city_idx) * 100) if len(city_idx) > 0 else 100.0
    
    # Overall Accuracy
    overall_accuracy = (age_accuracy + sal_accuracy + date_accuracy + city_accuracy) / 4
    
    print("EVALUATION RESULTS:")
    print(f" [o] Age Imputation Accuracy:     {age_accuracy:.2f}% (Average Error: {avg_age_error:.1f} years)")
    print(f" [o] Salary Imputation Accuracy:  {sal_accuracy:.2f}% (Average Error: {avg_sal_error:.1f})")
    print(f" [o] Date Standardization Acc:    {date_accuracy:.2f}%")
    print(f" [o] Text Normalization (Fuzzy):  {city_accuracy:.2f}%")
    print("------------------------------------------------------------------")
    print(f" SUCCESS RATE: {overall_accuracy:.2f}%")
    print("==================================================================")

if __name__ == "__main__":
    calculate_accuracy()
