import pandas as pd
import numpy as np
from backend.tools.synthetic_data.engine import (
    suggest_schema_from_prompt,
    compute_dcr,
    generate_report,
    generate_privacy_report,
    generate_data_dictionary
)

# 1. Test suggest_schema_from_prompt
print("--- Testing suggest_schema_from_prompt (Arabic) ---")
schema_ar = suggest_schema_from_prompt("جدول للطلاب مع درجاتهم وهواتفهم", 4, "ar_SA")
print("Arabic Schema Columns:")
for c in schema_ar:
    print(c)

print("\\n--- Testing suggest_schema_from_prompt (English) ---")
schema_en = suggest_schema_from_prompt("Customer transactions with amounts", 3, "en_US")
print("English Schema Columns:")
for c in schema_en:
    print(c)

# 2. Test compute_dcr and reports
print("\\n--- Testing compute_dcr and reports ---")
# Create simple original and synthetic DataFrames
orig_df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "age": [20, 25, 30, 35, 40],
    "salary": [5000, 6000, 7000, 8000, 9000],
    "city": ["Riyadh", "Jeddah", "Riyadh", "Dammam", "Makkah"]
})

synth_df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "age": [21, 24, 29, 36, 39],
    "salary": [5100, 5900, 7100, 7900, 9100],
    "city": ["Riyadh", "Jeddah", "Riyadh", "Dammam", "Makkah"]
})

profile = {
    "id": {"type": "id"},
    "age": {"type": "numerical", "min": 18, "max": 65, "mean": 30.0},
    "salary": {"type": "numerical", "min": 3000, "max": 12000, "mean": 7000.0},
    "city": {"type": "categorical", "value_counts": {"Riyadh": 2, "Jeddah": 1, "Dammam": 1, "Makkah": 1}}
}

# Run fidelity report
print("Fidelity Report:")
rep = generate_report(orig_df, synth_df, profile)
print(rep)

# Run DCR computation
print("\\nPrivacy DCR:")
dcr_results = compute_dcr(orig_df, synth_df, profile)
print("Mean DCR:", dcr_results["mean_dcr"])
print("Privacy Score:", dcr_results["privacy_score"])
print("Risk Level:", dcr_results["risk_level"])

# Run generate_privacy_report
print("\\nPrivacy Report DataFrame:")
df_priv, priv_dict = generate_privacy_report(orig_df, synth_df, profile)
print(df_priv)

# Run generate_data_dictionary
print("\\nData Dictionary Markdown:")
md_dict = generate_data_dictionary(profile, orig_df)
print(md_dict)

print("All tests completed successfully!")
