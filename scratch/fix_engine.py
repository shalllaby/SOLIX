import os

file_path = "backend/tools/synthetic_data/engine.py"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Verify boundaries
start_marker = "def inject_noise("
end_marker = "def generate_data_via_code_agent("

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print(f"Error: start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

print(f"Found inject_noise at index: {start_idx}")
print(f"Found generate_data_via_code_agent at index: {end_idx}")

# Define the clean replacement code including generate_privacy_report and generate_data_dictionary
clean_code = """def inject_noise(df: pd.DataFrame, profile: dict, null_pct: float = 0.0, outlier_pct: float = 0.0) -> pd.DataFrame:
    \"\"\"
    يقوم بحقن ضوضاء وقيم شاذة في البيانات المولدة بقصد تدريب النماذج على بيانات غير مثالية.
    null_pct: نسبة القيم المفقودة المراد إضافتها عشوائياً.
    outlier_pct: نسبة القيم الشاذة (Outliers) المراد حقنها في الأعمدة الرقمية.
    \"\"\"
    noisy_df = df.copy()
    n_rows = len(df)
    
    # 1. Null Injection (حقن القيم المفقودة)
    if null_pct > 0:
        for col in noisy_df.columns:
            if profile.get(col, {}).get("type") != "sensitive":
                n_nulls = int(n_rows * (null_pct / 100.0))
                if n_nulls > 0:
                    null_indices = np.random.choice(n_rows, n_nulls, replace=False)
                    noisy_df.loc[null_indices, col] = np.nan
                    
    # 2. Outlier Injection (حقن القيم المتطرفة للأعمدة الرقمية فقط)
    if outlier_pct > 0:
        num_cols = [c for c, v in profile.items() if isinstance(v, dict) and v.get("type") == "numerical"]
        for col in num_cols:
            valid_indices = noisy_df[noisy_df[col].notna()].index
            if len(valid_indices) == 0:
                continue
                
            n_outliers = int(len(valid_indices) * (outlier_pct / 100.0))
            if n_outliers > 0:
                outlier_indices = np.random.choice(valid_indices, n_outliers, replace=False)
                multipliers = np.random.choice([5, 10, -5], size=n_outliers)
                noisy_df.loc[outlier_indices, col] = noisy_df.loc[outlier_indices, col] * multipliers
                
    return noisy_df


# ─────────────────────────────────────────────
# 3. SIMILARITY REPORT  (تقرير التشابه)
# ─────────────────────────────────────────────

def fidelity_score_column(original: pd.Series, synthetic: pd.Series, col_type: str) -> float:
    \"\"\"
    يحسب نسبة التشابه لعمود واحد (0 → 100%).
    - الرقمي:   KS Test
    - الفئوي:  مقارنة توزيع التكرار
    - الحساس:  دايماً 100% (بيانات مولدة مستقلة)
    \"\"\"
    if col_type == "sensitive":
        return 100.0

    if col_type == "numerical":
        from scipy.stats import ks_2samp
        orig_clean = original.dropna()
        synt_clean = synthetic.dropna()
        if len(orig_clean) == 0 or len(synt_clean) == 0:
            return 0.0
        stat, _ = ks_2samp(orig_clean, synt_clean)
        return round((1 - stat) * 100, 2)

    if col_type == "categorical":
        orig_dist = original.value_counts(normalize=True)
        synt_dist = synthetic.value_counts(normalize=True)
        all_cats  = set(orig_dist.index) | set(synt_dist.index)
        diff = sum(abs(orig_dist.get(c, 0) - synt_dist.get(c, 0)) for c in all_cats)
        return round((1 - diff / 2) * 100, 2)

    return 0.0


def generate_report(
    original:  pd.DataFrame,
    synthetic: pd.DataFrame,
    profile:   dict
) -> pd.DataFrame:
    \"\"\"
    يرجع DataFrame يوضح Fidelity Score لكل عمود + المتوسط الكلي.
    \"\"\"
    rows = []
    for col in original.columns:
        if col not in synthetic.columns:
            continue
        if col not in profile or not isinstance(profile.get(col), dict):
            continue
        col_type = profile[col]["type"]
        score    = fidelity_score_column(original[col], synthetic[col], col_type)
        rows.append({
            "column":        col,
            "type":         col_type,
            "fidelity_score": score,
            "quality":        "ممتاز 🟢" if score >= 85 else "مقبول 🟡" if score >= 60 else "ضعيف 🔴"
        })

    report_df = pd.DataFrame(rows)
    return report_df


# ─────────────────────────────────────────────
# 4. PRIVACY ANALYSIS  (تحليل الخصوصية)
# ─────────────────────────────────────────────

def _prepare_numeric_matrix(df: pd.DataFrame, profile: dict) -> np.ndarray:
    \"\"\"
    يحول DataFrame إلى مصفوفة رقمية للمقارنة.
    - الأعمدة الرقمية: يتم تطبيعها (Normalize) بين 0 و 1.
    - الأعمدة الفئوية: يتم تحويلها إلى أرقام (Label Encoding).
    - الأعمدة الحساسة: يتم تجاهلها (لأنها مولدة بـ Faker ولا معنى لمقارنتها).
    \"\"\"
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    cols_to_use = []
    encoded_parts = []

    for col in df.columns:
        if col not in profile or not isinstance(profile[col], dict):
            continue
        col_type = profile[col].get("type", "")

        if col_type == "sensitive":
            continue

        if col_type == "numerical":
            cols_to_use.append(col)
            vals = df[col].fillna(0).values.reshape(-1, 1)
            scaler = MinMaxScaler()
            encoded_parts.append(scaler.fit_transform(vals))

        elif col_type == "categorical":
            cols_to_use.append(col)
            le = LabelEncoder()
            vals = le.fit_transform(df[col].fillna("__MISSING__").astype(str))
            if vals.max() > 0:
                vals = vals / vals.max()
            encoded_parts.append(vals.reshape(-1, 1))

    if not encoded_parts:
        return np.array([])

    return np.hstack(encoded_parts)


def compute_dcr(original: pd.DataFrame, synthetic: pd.DataFrame, profile: dict, sample_size: int = 500) -> dict:
    \"\"\"
    يحسب Distance to Closest Record (DCR) لكل صف اصطناعي.
    \"\"\"
    from scipy.spatial.distance import cdist

    orig_matrix = _prepare_numeric_matrix(original, profile)
    synth_matrix = _prepare_numeric_matrix(synthetic, profile)

    if orig_matrix.size == 0 or synth_matrix.size == 0:
        return {
            "dcr_values": [],
            "mean_dcr": 0, "min_dcr": 0, "max_dcr": 0,
            "privacy_score": 0, "risk_level": "غير قابل للحساب",
            "at_risk_count": 0, "at_risk_pct": 0
        }

    if len(synth_matrix) > sample_size:
        idx = np.random.choice(len(synth_matrix), sample_size, replace=False)
        synth_sample = synth_matrix[idx]
    else:
        synth_sample = synth_matrix

    distances = cdist(synth_sample, orig_matrix, metric='euclidean')
    dcr_values = distances.min(axis=1)

    mean_dcr = float(np.mean(dcr_values))
    min_dcr = float(np.min(dcr_values))
    max_dcr = float(np.max(dcr_values))

    risk_threshold = mean_dcr * 0.15
    at_risk = (dcr_values < risk_threshold).sum()
    at_risk_pct = round((at_risk / len(dcr_values)) * 100, 2)
    privacy_score = round(max(0, 100 - at_risk_pct), 2)

    if privacy_score >= 95:
        risk_level = "آمن تماماً 🟢"
    elif privacy_score >= 80:
        risk_level = "آمن 🟢"
    elif privacy_score >= 60:
        risk_level = "مقبول 🟡"
    elif privacy_score >= 40:
        risk_level = "محفوف بالمخاطر 🟠"
    else:
        risk_level = "خطر تسريب عالٍ 🔴"

    return {
        "dcr_values": dcr_values,
        "mean_dcr": round(mean_dcr, 4),
        "min_dcr": round(min_dcr, 4),
        "max_dcr": round(max_dcr, 4),
        "privacy_score": privacy_score,
        "risk_level": risk_level,
        "at_risk_count": int(at_risk),
        "at_risk_pct": at_risk_pct,
        "risk_threshold": round(risk_threshold, 4),
        "total_checked": len(dcr_values)
    }


def generate_privacy_report(original: pd.DataFrame, synthetic: pd.DataFrame, profile: dict) -> tuple:
    \"\"\"
    يولد تقرير خصوصية مفصل بجدول يمكن عرضه في الواجهة.
    \"\"\"
    privacy = compute_dcr(original, synthetic, profile)

    rows = [
        {"المقياس": "عدد الصفوف التي تم فحصها",         "القيمة": str(privacy["total_checked"])},
        {"المقياس": "متوسط المسافة (Mean DCR)",           "القيمة": str(privacy["mean_dcr"])},
        {"المقياس": "أقل مسافة (Min DCR) — أخطر صف",     "القيمة": str(privacy["min_dcr"])},
        {"المقياس": "أعلى مسافة (Max DCR) — أكثر أماناً", "القيمة": str(privacy["max_dcr"])},
        {"المقياس": "حد الخطر (Risk Threshold)",          "القيمة": str(privacy["risk_threshold"])},
        {"المقياس": "عدد الصفوف المعرضة للخطر",           "القيمة": str(privacy["at_risk_count"])},
        {"المقياس": "نسبة الصفوف المعرضة للخطر",          "القيمة": f"{privacy['at_risk_pct']}%"},
        {"المقياس": "─── نسبة الخصوصية الكلية ───",      "القيمة": f"{privacy['privacy_score']}%"},
        {"المقياس": "─── مستوى الأمان ───",               "القيمة": privacy["risk_level"]},
    ]

    return pd.DataFrame(rows), privacy


def generate_data_dictionary(profile: dict, df: pd.DataFrame) -> str:
    \"\"\"يولد نص Markdown يحتوي على قاموس مفصل لجميع الأعمدة في البيانات.\"\"\"
    md = "# 📚 قاموس البيانات (Data Dictionary)\\n\\n"
    md += f"**إجمالي الصفوف:** {len(df)}\\n"
    md += f"**إجمالي الأعمدة:** {len(df.columns)}\\n\\n"
    md += "---\\n\\n"
    for col, info in profile.items():
        if col.startswith("__"): continue
        md += f"### 🔹 عمود: `{col}`\\n"
        md += f"- **النوع (Type):** `{info.get('type', 'Unknown')}`\\n"
        md += f"- **نسبة القيم المفقودة (Nulls):** {info.get('null_pct', 0):.2f}%\\n"
        if info.get('type') == 'numerical':
            md += f"- **الحد الأدنى (Min):** {info.get('min', 'N/A')}\\n"
            md += f"- **الحد الأقصى (Max):** {info.get('max', 'N/A')}\\n"
            md += f"- **المتوسط (Mean):** {info.get('mean', 'N/A'):.2f}\\n"
        elif info.get('type') == 'categorical':
            val_counts = info.get('value_counts', {})
            top_vals = list(val_counts.keys())[:5]
            md += f"- **أبرز الفئات المتكررة:** {', '.join([str(x) for x in top_vals])}\\n"
        md += "\\n"
    return md


def suggest_schema_from_prompt(user_prompt: str, num_columns: int = 5, locale: str = "en_US") -> list:
    \"\"\"
    يقترح هيكل بيانات (Schema) بناءً على وصف المستخدم والـ locale المطلوبة.
    \"\"\"
    import json
    import threading
    import os

    target_language = "Arabic" if locale.startswith("ar") else "English"

    prompt = f\"\"\"أنت خبير علم بيانات ومبرمج بايثون محترف للغاية.
المطلوب منك هو اقتراح هيكل بيانات (Schema) لجدول يحتوي على {num_columns} أعمدة يلائم وصف المستخدم التالي:
\\\"{user_prompt}\\\"

اللغة/الدولة المستهدفة (locale): {locale}

الأنواع المدعومة (type) التي يجب عليك الاختيار منها حصراً هي:
1. \\\"id\\\": رقم تسلسلي أو معرّف فريد.
2. \\\"name\\\": اسم شخص كامل.
3. \\\"email\\\": بريد إلكتروني.
4. \\\"phone\\\": رقم هاتف متوافق مع سياق الوصف والدولة.
5. \\\"address\\\": عنوان جغرافي.
6. \\\"number\\\": أرقام وحسابات (مثال: السعر، العمر، الكمية). حدد \\\"min\\\" و \\\"max\\\" و \\\"is_float\\\" في التفاصيل.
7. \\\"category\\\": فئات/تصنيفات خيارات محددة. أرفق قائمة بالخيارات المقترحة \\\"categories\\\" مثلاً: [\\\"رياضة\\\", \\\"ألعاب\\\", \\\"موسيقى\\\"].
8. \\\"date\\\": تاريخ (مثلاً تاريخ ميلاد، تاريخ شراء).
9. \\\"text\\\": نص حر قصير (مثلاً ملاحظات، وصف منتج).

صيغة الإرجاع يجب أن تكون كائن JSON فقط يحتوي على قائمة من الأعمدة بالصيغة التالية:
{{
  \\\"columns\\\": [
    {{
      \\\"name\\\": \\\"column_name_in_english\\\",
      \\\"display_name\\\": \\\"اسم العمود باللغة المطلوبة\\\",
      \\\"type\\\": \\\"type_name\\\",
      \\\"description\\\": \\\"وصف مختصر للعمود باللغة المطلوبة\\\",
      \\\"min\\\": 10,  // اختياري فقط إذا كان النوع number
      \\\"max\\\": 100, // اختياري فقط إذا كان النوع number
      \\\"is_float\\\": false, // اختياري فقط إذا كان النوع number
      \\\"categories\\\": [\\\"cat1\\\", \\\"cat2\\\"] // إجباري فقط إذا كان النوع category
    }}
  ]
}}

تعليمات صارمة جداً بشأن تطابق اللغة والمحلية (Locale & Language Consistency):
- The user requested the locale '{locale}'. Therefore, the target language for this dataset is {target_language}.
- You MUST generate all values for 'display_name', 'description', and 'categories' strictly and entirely in {target_language}.
- NO LANGUAGE MIXING: If {target_language} is Arabic, under no circumstances should you return English words inside the 'categories', 'display_name', or 'description' fields.
- The ONLY exception is the programmatic 'name' field, which must always remain in English snake_case (e.g., 'car_color').
\"\"\"

    result_container = {"result": None, "error": None}

    def _call_llm():
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY", "").strip() or BACKUP_GROQ_API_KEY
            if not key.startswith("gsk"):
                raise ValueError("No valid API key")
            client = Groq(api_key=key)
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                result_container["result"] = json.loads(resp.choices[0].message.content)
            except Exception as first_err:
                err_msg = str(first_err)
                if "429" in err_msg or "rate limit" in err_msg.lower() or "limit exceeded" in err_msg.lower() or "too many requests" in err_msg.lower():
                    print('INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"')
                    backup_model = "meta-llama/llama-4-scout-17b-16e-instruct"
                    print(f"Switching automatically to backup model: {backup_model}")
                    resp = client.chat.completions.create(
                        model=backup_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=4096,
                        response_format={"type": "json_object"},
                    )
                    result_container["result"] = json.loads(resp.choices[0].message.content)
                else:
                    raise first_err
        except Exception as e:
            result_container["error"] = str(e)

    # تشغيل في Thread مع Timeout 30 ثانية
    t = threading.Thread(target=_call_llm, daemon=True)
    t.start()
    t.join(timeout=30)

    # معالجة النتائج أو استخدام الفولباك
    columns = []
    if result_container["result"] is not None and "columns" in result_container["result"]:
        columns = result_container["result"]["columns"]
    else:
        # Fallback schema
        is_ar = locale.startswith("ar")
        if is_ar:
            columns = [
                {"name": "id", "display_name": "المعرف", "type": "id", "description": "Serial ID"},
                {"name": "name", "display_name": "الاسم الكامل", "type": "name", "description": "Full name"},
                {"name": "email", "display_name": "البريد الإلكتروني", "type": "email", "description": "Email address"},
                {"name": "phone", "display_name": "رقم الهاتف", "type": "phone", "description": "Phone number"},
                {"name": "city", "display_name": "المدينة", "type": "category", "description": "City", "categories": ["الرياض", "جدة", "الدمام", "مكة"]},
                {"name": "age", "display_name": "العمر", "type": "number", "description": "Age", "min": 18, "max": 65, "is_float": False}
            ]
        else:
            columns = [
                {"name": "id", "display_name": "ID", "type": "id", "description": "Serial ID"},
                {"name": "name", "display_name": "Name", "type": "name", "description": "Full name"},
                {"name": "email", "display_name": "Email", "type": "email", "description": "Email address"},
                {"name": "phone", "display_name": "Phone", "type": "phone", "description": "Phone number"},
                {"name": "country", "display_name": "Country", "type": "category", "description": "Country category", "categories": ["USA", "UK", "Canada", "Germany"]},
                {"name": "salary", "display_name": "Salary", "type": "number", "description": "Salary amount", "min": 3000, "max": 12000, "is_float": False}
            ]

    # تعقيم وتأكيد تطابق الأنواع المدعومة
    allowed_types = {"id", "name", "email", "phone", "address", "number", "category", "date", "text"}
    sanitized_columns = []
    for col in columns:
        col_type = col.get("type", "text").lower()
        if col_type not in allowed_types:
            col_type = "text"
        col["type"] = col_type
        sanitized_columns.append(col)

    return sanitized_columns


"""

# Combine and write
new_content = content[:start_idx] + clean_code + content[end_idx:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Replacement done successfully!")
