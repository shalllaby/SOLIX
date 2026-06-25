def generate_data_dictionary(profile: dict, df: pd.DataFrame) -> str:
    md = "# 📚 قاموس البيانات (Data Dictionary)\n\n"
    md += f"**عدد الصفوف:** {len(df)}\n\n"
    md += f"**عدد الأعمدة:** {len(df.columns)}\n\n"
    md += "---\n\n"
    for col, info in profile.items():
        if col.startswith("__"): continue
        md += f"### 🔹 {col}\n"
        md += f"- **النوع (Type):** `{info.get('type', 'Unknown')}`\n"
        md += f"- **النسبة المفقودة (Nulls):** {info.get('null_pct', 0):.2f}%\n"
        if info.get('type') == 'numerical':
            md += f"- **الحد الأدنى (Min):** {info.get('min', 'N/A')}\n"
            md += f"- **الحد الأقصى (Max):** {info.get('max', 'N/A')}\n"
            md += f"- **المتوسط (Mean):** {info.get('mean', 'N/A'):.2f}\n"
        elif info.get('type') == 'categorical':
            val_counts = info.get('value_counts', {})
            top_vals = list(val_counts.keys())[:5]
            md += f"- **أبرز الفئات (Top Categories):** {', '.join([str(x) for x in top_vals])}\n"
        md += "\n"
    return md
