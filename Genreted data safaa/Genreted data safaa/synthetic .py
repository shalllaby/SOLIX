"""
SOL Platform — Synthetic Data Generation Engine
الموديول الأساسي: تحليل + توليد + تقرير
يدعم طريقتين: الإحصائية (Basic) والذكاء الاصطناعي (CTGAN)
"""

import pandas as pd
import numpy as np
from faker import Faker
from scipy.stats import ks_2samp
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import os
warnings.filterwarnings("ignore")

fake = Faker()

# 🔑 المفتاح الأساسي لتشغيل ذكاء الفهم العميق في الخلفية
GROQ_API_KEY = "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"

# ─── Cache داخلي لنتائج LLM (يمنع إعادة الاستدعاء للملف نفسه) ───
_LLM_CACHE: dict = {}


# ─────────────────────────────────────────────
# 1. AI-DRIVEN STATISTICAL PROFILING  (تحليل الهيكل بالفهم العميق)
# ─────────────────────────────────────────────

def _col_signature(df: pd.DataFrame) -> str:
    """بصمة فريدة للأعمدة تُستخدم كمفتاح Cache — سريعة جداً."""
    parts = []
    for col in df.columns[:50]:  # نأخذ أول 50 عمود كحد أقصى
        first_val = str(df[col].iloc[0]) if len(df) > 0 else ""
        parts.append(f"{col}:{df[col].dtype}:{first_val[:20]}")
    return "|".join(parts)


def _fallback_types(df: pd.DataFrame) -> dict:
    """كشف أنواع الأعمدة بدون LLM — فوري (Fallback) مع قواعد استبعاد ذكية للأعمدة غير البشرية."""
    result = {}
    SENSITIVE_KW = ["name", "email", "phone", "address", "ssn", "اسم", "بريد", "هاتف", "عنوان"]
    NON_HUMAN_KW = [
        "kepler", "kepoi", "product", "file", "dataset", "class", "category", "model", "type", 
        "country", "city", "state", "town", "planet", "star", "galaxy", "hash", "id", "code", 
        "key", "symbol", "system", "object", "app", "application", "device", "host", "domain", 
        "site", "url", "path", "dir", "directory", "table", "column", "field", "row", "index", 
        "status", "group", "team", "role", "permission", "action", "event", "log", "error", 
        "warn", "info", "debug", "trace", "version", "tag", "label", "title", "subject", 
        "topic", "level", "rank", "grade", "score", "value", "format", "ext", "extension"
    ]

    for col in df.columns:
        col_lower = col.lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "numerical"
            continue

        if any(kw in col_lower for kw in SENSITIVE_KW):
            if "email" in col_lower or "بريد" in col_lower:
                result[col] = "sensitive_email"
            elif "phone" in col_lower or "هاتف" in col_lower:
                result[col] = "sensitive_phone"
            elif "address" in col_lower or "عنوان" in col_lower:
                result[col] = "sensitive_address"
            elif "ssn" in col_lower:
                result[col] = "sensitive_id"
            else:
                # التحقق مما إذا كان الاسم غير بشري بناءً على الكلمات المفتاحية المستبعدة
                has_exclusion = any(ex in col_lower for ex in NON_HUMAN_KW)
                
                # التحقق من محتوى البيانات الفعلي (إذا كان يحتوي على أرقام بنسبة > 5% فليس اسماً بشرياً)
                contains_digits = False
                non_null_vals = df[col].dropna().head(100).astype(str)
                if len(non_null_vals) > 0:
                    digit_matches = non_null_vals.str.contains(r'\d').sum()
                    if (digit_matches / len(non_null_vals)) > 0.05:
                        contains_digits = True
                
                if has_exclusion or contains_digits:
                    result[col] = "categorical"
                else:
                    result[col] = "sensitive_name"
        else:
            result[col] = "categorical"
    return result


def detect_column_types_llm(df: pd.DataFrame) -> dict:
    """
    يستخدم Groq LLM لفهم عينة من البيانات وإرجاع نوع كل عمود.
    ميزات الأداء:
    - Cache: لو نفس الأعمدة + نفس القيم → يرجع النتيجة فوراً بدون API call
    - Timeout: 10 ثوانٍ حد أقصى، بعدها يستخدم Fallback تلقائياً
    - حد الأعمدة: يرسل 40 عمود فقط للـ LLM (أسرع وأرخص)
    """
    import threading

    # ── 1. تحقق من الـ Cache أولاً ──
    cache_key = _col_signature(df)
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    # ── 2. تجهيز العينة المرسلة للـ LLM ──
    MAX_COLS_FOR_LLM = 40
    cols_to_send = list(df.columns[:MAX_COLS_FOR_LLM])
    sample_csv = df[cols_to_send].head(3).to_csv(index=False)  # 3 صفوف فقط — أسرع

    prompt = f"""أنت خبير بيانات. حدد نوع كل عمود من البيانات التالية.
لا تخلط بين الأرقام الحسابية والبيانات الحساسة.
ملاحظة هامة جداً: الأعمدة العلمية أو التقنية أو أسماء المنتجات والملفات (مثل kepler_name، product_name) التي لا تخص أسماء أشخاص يجب تصنيفها كـ "categorical" وليس "sensitive_name".

بيانات (CSV):
{sample_csv}

أنواع مسموح بها حصراً:
"numerical" | "categorical" | "sensitive_name" | "sensitive_email" | "sensitive_phone" | "sensitive_address" | "sensitive_id"

أرجع JSON فقط: {{"col_name": "type", ...}}"""

    result_container = {"result": None, "error": None}

    def _call_llm():
        try:
            from groq import Groq
            key = GROQ_API_KEY if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk") else os.getenv("GROQ_API_KEY", "")
            if not key.startswith("gsk"):
                raise ValueError("No valid API key")
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            result_container["result"] = json.loads(resp.choices[0].message.content)
        except Exception as e:
            result_container["error"] = str(e)

    # ── 3. تشغيل الـ LLM في Thread مع Timeout 10 ثوانٍ ──
    t = threading.Thread(target=_call_llm, daemon=True)
    t.start()
    t.join(timeout=10)  # ← أقصى انتظار 10 ثوانٍ

    raw_result = {}
    if result_container["result"] is not None:
        raw_result = result_container["result"]
        # لو فيه أعمدة مش في الـ LLM result (أكثر من 40) → نكملها بالـ Fallback
        if len(df.columns) > MAX_COLS_FOR_LLM:
            fb = _fallback_types(df)
            for col in df.columns:
                if col not in raw_result:
                    raw_result[col] = fb.get(col, "categorical")
    else:
        # Fallback: LLM تجاوز الـ Timeout أو فشل
        err = result_container.get("error", "timeout")
        print(f"LLM profiling fallback (reason: {err})")
        raw_result = _fallback_types(df)

    # ── 5. فلترة وحماية للأسماء غير البشرية (Sanity Check) ──
    # للتأكد بنسبة 100% أن أي عمود تم تصنيفه كـ sensitive_name ليس اسماً علمياً أو تقنياً
    NON_HUMAN_KW = [
        "kepler", "kepoi", "product", "file", "dataset", "class", "category", "model", "type", 
        "country", "city", "state", "town", "planet", "star", "galaxy", "hash", "id", "code", 
        "key", "symbol", "system", "object", "app", "application", "device", "host", "domain", 
        "site", "url", "path", "dir", "directory", "table", "column", "field", "row", "index", 
        "status", "group", "team", "role", "permission", "action", "event", "log", "error", 
        "warn", "info", "debug", "trace", "version", "tag", "label", "title", "subject", 
        "topic", "level", "rank", "grade", "score", "value", "format", "ext", "extension"
    ]
    
    final_result = {}
    for col, col_type in raw_result.items():
        col_lower = col.lower()
        if col_type == "sensitive_name":
            has_exclusion = any(ex in col_lower for ex in NON_HUMAN_KW)
            contains_digits = False
            non_null_vals = df[col].dropna().head(100).astype(str)
            if len(non_null_vals) > 0:
                digit_matches = non_null_vals.str.contains(r'\d').sum()
                if (digit_matches / len(non_null_vals)) > 0.05:
                    contains_digits = True
            
            if has_exclusion or contains_digits:
                final_result[col] = "categorical"
                continue
                
        final_result[col] = col_type

    _LLM_CACHE[cache_key] = final_result
    return final_result


# ─────────────────────────────────────────────
# TEMPLATE PATTERN EXTRACTION (استخراج أنماط النصوص)
# ─────────────────────────────────────────────

def extract_pattern_template(series: pd.Series) -> str:
    """
    تحلل العمود وتستخرج نمطاً (Template) متوافقاً مع Faker bothify.
    مثال: K00752.01 -> K#####.##
    Kepler-227 b -> Kepler-### ?
    """
    # تنظيف وتصفية القيم غير الفارغة
    vals = series.dropna().astype(str).head(100)
    if len(vals) == 0:
        return ""
        
    # 1. إيجاد البادئة المشتركة (Common Prefix)
    first_val = vals.iloc[0]
    prefix = ""
    for i in range(len(first_val)):
        char = first_val[i]
        if all(v.startswith(prefix + char) for v in vals):
            prefix += char
        else:
            break
            
    # 2. إيجاد اللاحقة المشتركة (Common Suffix)
    suffix = ""
    for i in range(1, len(first_val) + 1):
        char = first_val[-i]
        if all(v.endswith(char + suffix) for v in vals):
            suffix = char + suffix
        else:
            break
            
    # إذا كانت البادئة أو اللاحقة تغطي النص بالكامل، نقوم بضبطها
    if len(prefix) + len(suffix) >= len(first_val):
        suffix = ""
        
    # 3. تحليل الجزء المتغير وتحويله إلى نمط (# للأرقام، ? للحروف، وغيرها يبقى كما هو)
    templates = []
    for v in vals:
        middle = v
        if prefix:
            middle = middle[len(prefix):]
        if suffix:
            middle = middle[:-len(suffix)]
            
        t_chars = []
        for char in middle:
            if char.isdigit():
                t_chars.append("#")
            elif char.isalpha():
                t_chars.append("?")
            else:
                t_chars.append(char)
        templates.append("".join(t_chars))
        
    if not templates:
        return ""
        
    from collections import Counter
    most_common_template = Counter(templates).most_common(1)[0][0]
    return prefix + most_common_template + suffix


# ─────────────────────────────────────────────
# PROFILING — تحليل كامل للملف بأكمله (Pandas سريع جداً)
# ─────────────────────────────────────────────

# حد عينة تدريب CTGAN فقط (ليس للتحليل)
CTGAN_TRAIN_SAMPLE = 2_000   # CTGAN يتعلم التوزيع، 2,000 صف كافٍ تماماً

def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    يدرس كل عمود ويرجع قاموس بخصائصه الإحصائية مدمجاً بذكاء LLM للفهم.
    يحلّل الملف كاملاً بدون عينة — عمليات Pandas متجهيزة وسريعة لأي حجم.
    تحليل LLM يعتمد على 3 صفوف فقط + Cache → سريع جداً.
    """
    import re

    profile = {}

    # حفظ معلومات عامة عن الملف
    profile["__meta__"] = {
        "total_rows":    len(df),
        "analyzed_rows": len(df),   # دائماً كامل
        "was_sampled":   False,
    }

    # LLM: يرسل 3 صفوف فقط ولديه Cache → فوري بعد المرة الأولى
    llm_types = detect_column_types_llm(df)

    # ── حساب الإحصائيات المجمّعة مسبقاً بعملية واحدة لكل عمود (Vectorized) ──
    null_pcts   = df.isnull().mean() * 100   # O(n) مرة واحدة
    num_uniques = df.nunique()               # O(n) مرة واحدة

    for col in df.columns:
        col_type = llm_types.get(col, "categorical")

        # حماية: numerical والعمود مش أرقام فعلياً
        if col_type == "numerical" and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                col_type = "categorical"

        # حماية: أرقام لا تتحول إلى sensitive أبداً
        if col_type.startswith("sensitive") and pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numerical"

        # ── كشف ID/أرقام تسلسلية ──
        is_id     = False
        prefix    = ""
        start_num = 1
        n_unique  = num_uniques[col]
        n_total   = len(df)

        if pd.api.types.is_integer_dtype(df[col]) and n_unique == n_total:
            is_id     = True
            start_num = int(df[col].min())

        elif n_unique == n_total and df[col].dtype == object:
            sample_vals = df[col].dropna().head(10).astype(str)
            if len(sample_vals) > 0:
                m = re.match(r'^([^0-9]+)(\d+)$', sample_vals.iloc[0])
                if m:
                    pot_prefix = m.group(1)
                    all_match = all(
                        str(v).startswith(pot_prefix) and str(v)[len(pot_prefix):].isdigit()
                        for v in sample_vals
                    )
                    if all_match:
                        is_id     = True
                        prefix    = pot_prefix
                        nums      = df[col].dropna().astype(str).str.replace(pot_prefix, '', regex=False).astype(int)
                        start_num = int(nums.min())

        if is_id:
            col_type = "id_sequence"

        # ── كشف أعمدة الأنماط عالية التكرار (Pattern Sequence) ──
        template = ""
        if col_type == "categorical":
            if n_unique > 20 and (n_unique / n_total) > 0.05:
                template = extract_pattern_template(df[col])
                if template and ("#" in template or "?" in template):
                    col_type = "pattern_sequence"

        # ── تجميع معلومات العمود ──
        info = {"type": col_type, "null_pct": null_pcts[col]}

        if col_type == "numerical":
            info.update({
                "mean":   float(df[col].mean()),
                "std":    float(df[col].std()),
                "min":    float(df[col].min()),
                "max":    float(df[col].max()),
                "is_int": pd.api.types.is_integer_dtype(df[col]),
            })
        elif col_type == "categorical":
            info["value_counts"] = df[col].value_counts(normalize=True).to_dict()
        elif col_type == "pattern_sequence":
            info["template"] = template
        elif col_type == "id_sequence":
            info["min"]    = start_num
            info["prefix"] = prefix

        profile[col] = info

    # الارتباطات بين الأعمدة الرقمية (Vectorized — سريع)
    num_cols = [c for c, v in profile.items() if isinstance(v, dict) and v.get("type") == "numerical"]
    if len(num_cols) > 1:
        profile["__correlations__"] = df[num_cols].corr().to_dict()

    return profile


# ─────────────────────────────────────────────
# 2A. BASIC GENERATION ENGINE  (توليد إحصائي — مع Threads)
# ─────────────────────────────────────────────

def _generate_numerical(info: dict, n: int) -> pd.Series:
    """يولد أرقاماً تحافظ على المتوسط والمدى."""
    values = np.random.normal(loc=info["mean"], scale=info["std"] if info["std"] > 0 else 1, size=n)
    values = np.clip(values, info["min"], info["max"])
    if info.get("is_int"):
        values = np.round(values).astype(int)
    return pd.Series(values)


def _generate_categorical(info: dict, n: int) -> pd.Series:
    """يولد قيم فئوية بنفس نسب التكرار الأصلية."""
    categories = list(info["value_counts"].keys())
    weights    = list(info["value_counts"].values())
    return pd.Series(np.random.choice(categories, size=n, p=weights))


def _generate_sensitive(info: dict, n: int) -> pd.Series:
    """يولد بيانات حساسة وهمية واقعية باستخدام Faker بناءً على التوجيه الدقيق من LLM."""
    col_type = info["type"]
    if col_type == "sensitive_email":
        values = [fake.email() for _ in range(n)]
    elif col_type == "sensitive_phone":
        values = [fake.phone_number() for _ in range(n)]
    elif col_type == "sensitive_address":
        values = [fake.address().replace("\n", ", ") for _ in range(n)]
    elif col_type == "sensitive_id":
        values = [fake.ssn() for _ in range(n)]
    else: # sensitive_name أو أي نوع آخر
        values = [fake.name() for _ in range(n)]
    return pd.Series(values)


def _generate_single_column(args):
    """دالة مساعدة لتوليد عمود واحد — تُستخدم داخل ThreadPoolExecutor."""
    col, info, n_rows = args
    if info["type"] == "numerical":
        return col, _generate_numerical(info, n_rows)
    elif info["type"] == "categorical":
        return col, _generate_categorical(info, n_rows)
    elif info["type"].startswith("sensitive"):
        return col, _generate_sensitive(info, n_rows)
    elif info["type"] == "id_sequence":
        start_val = info.get("min", 1)
        orig_prefix = info.get("prefix", "")
        # إضافة بادئة 'SOL-' لتمييز أن هذا الصف اصطناعي مع الحفاظ على البادئة الأصلية (مثال: SOL-CUST_1000)
        return col, pd.Series([f"SOL-{orig_prefix}{i}" for i in range(start_val, start_val + n_rows)])
    elif info["type"] == "pattern_sequence":
        template = info["template"]
        generated_vals = set()
        attempts = 0
        required_unique = n_rows
        while len(generated_vals) < required_unique and attempts < required_unique * 3:
            generated_vals.add(fake.bothify(template))
            attempts += 1
        val_list = list(generated_vals)
        if len(val_list) < n_rows:
            val_list += [fake.bothify(template) for _ in range(n_rows - len(val_list))]
        return col, pd.Series(val_list[:n_rows])
    return col, pd.Series([None] * n_rows)



def generate_synthetic(df: pd.DataFrame, profile: dict, n_rows: int) -> pd.DataFrame:
    """
    يولد DataFrame اصطناعي بعدد الصفوف المطلوب (الطريقة الإحصائية الأساسية).
    يستخدم ThreadPoolExecutor لمعالجة الأعمدة بالتوازي لتحسين الأداء.
    """
    # تجهيز المهام — كل عمود مهمة مستقلة
    tasks = [(col, profile[col], n_rows) for col in df.columns if col in profile]

    synthetic = {}

    # تشغيل الأعمدة بالتوازي عبر Threads
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
        futures = {executor.submit(_generate_single_column, task): task[0] for task in tasks}
        for future in as_completed(futures):
            col, series = future.result()
            synthetic[col] = series

    # إعادة ترتيب الأعمدة بنفس ترتيب الملف الأصلي
    result = pd.DataFrame({col: synthetic[col] for col in df.columns if col in synthetic})

    # إضافة صفوف null بنفس النسبة الأصلية
    for col in df.columns:
        if col not in profile:
            continue
        null_pct = profile[col]["null_pct"]
        if null_pct > 0:
            null_mask = np.random.rand(n_rows) < (null_pct / 100)
            result.loc[null_mask, col] = np.nan

    return result


# ─────────────────────────────────────────────
# 2B. AI GENERATION ENGINE (CTGAN / SDV)
# ─────────────────────────────────────────────

def _optimize_metadata_for_sdv(metadata, df: pd.DataFrame, profile: dict):
    """
    تقوم بتعديل الـ Metadata الخاص بـ SDV لمنع تحويل الأعمدة النصية الفريدة (مثل kepoi_name) 
    إلى معرفات عشوائية (sdv-id-XXXXXX)، وتحويلها بدلاً من ذلك إلى فئات (categorical) 
    للحفاظ على شكل البيانات الحقيقية وتوزيعها.
    """
    meta_dict = metadata.to_dict()
    table_name = list(meta_dict.get('tables', {}).keys())[0] if meta_dict.get('tables') else 'table'
    columns = meta_dict.get('tables', {}).get(table_name, {}).get('columns', {})
    primary_key = meta_dict.get('tables', {}).get(table_name, {}).get('primary_key', None)
    
    cols_to_convert = []
    for col, col_info in columns.items():
        if col_info.get('sdtype') == 'id':
            col_type = profile.get(col, {}).get('type', 'categorical')
            if col_type != 'id_sequence':
                cols_to_convert.append(col)
                
    if cols_to_convert:
        if primary_key in cols_to_convert:
            metadata.remove_primary_key()
        for col in cols_to_convert:
            metadata.update_column(column_name=col, sdtype='categorical')
            
    return metadata


def generate_synthetic_ai(df: pd.DataFrame, profile: dict, n_rows: int, epochs: int = 30, train_sample_size: int = 2000, progress_callback=None) -> pd.DataFrame:
    """
    يولّد DataFrame اصطناعي باستخدام CTGAN (شبكات عصبية توليدية).
    للملفات الكبيرة: يتدرّب على عينة ذكية (الـ GAN يتعلّم التوزيع، لا يحفظ الصفوف).
    """
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata

    # حذف الأعمدة الحساسة + IDs + الأنماط الفريدة مؤقتاً
    excluded_cols = [
        c for c, v in profile.items()
        if not c.startswith("__") and (
            v.get("type", "").startswith("sensitive") or 
            v.get("type") == "id_sequence" or
            v.get("type") == "pattern_sequence"
        )
    ]
    df_clean = df.drop(columns=excluded_cols, errors="ignore")

    # ── عينة ذكية لتدريب CTGAN ──
    # CTGAN يتعلّم التوزيع المشترك بين الأعمدة، وليس بحاجة لكل الصفوف
    n_train = len(df_clean)
    train_df = df_clean
    sampled_for_training = False
    if train_sample_size and n_train > train_sample_size:
        train_df = df_clean.sample(n=train_sample_size, random_state=42)
        sampled_for_training = True

    if progress_callback:
        msg = (
            f"جاري تحليل البيانات وبناء الـ Metadata..."
            if not sampled_for_training else
            f"✅ تحليل {n_train:,} صف — تدريب CTGAN على عينة {train_sample_size:,} صف..."
        )
        progress_callback(0.1, msg)

    metadata = Metadata.detect_from_dataframe(data=train_df)
    metadata = _optimize_metadata_for_sdv(metadata, train_df, profile)

    if progress_callback:
        progress_callback(0.2, f"جاري تدريب نموذج CTGAN... ({epochs} epoch، عينة {len(train_df):,} صف)")

    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    synthesizer.fit(train_df)

    if progress_callback:
        progress_callback(0.7, "جاري توليد البيانات الاصطناعية...")

    synthetic_df = synthesizer.sample(num_rows=n_rows)

    if progress_callback:
        progress_callback(0.85, "جاري إضافة البيانات الحساسة الوهمية والأعمدة الفريدة...")

    for col in excluded_cols:
        col_type = profile[col]["type"]
        if col_type == "id_sequence":
            start_val  = profile[col].get("min", 1)
            orig_prefix = profile[col].get("prefix", "")
            synthetic_df[col] = [f"SOL-{orig_prefix}{i}" for i in range(start_val, start_val + n_rows)]
        elif col_type == "pattern_sequence":
            template = profile[col]["template"]
            generated_vals = set()
            attempts = 0
            required_unique = n_rows
            while len(generated_vals) < required_unique and attempts < required_unique * 3:
                generated_vals.add(fake.bothify(template))
                attempts += 1
            val_list = list(generated_vals)
            if len(val_list) < n_rows:
                val_list += [fake.bothify(template) for _ in range(n_rows - len(val_list))]
            synthetic_df[col] = val_list[:n_rows]
        else:
            synthetic_df[col] = _generate_sensitive(profile[col], n_rows)

    synthetic_df = synthetic_df[df.columns]

    if progress_callback:
        progress_callback(1.0, "تم الانتهاء!")

    return synthetic_df


# ─────────────────────────────────────────────
# 2B_2. FAST AI ENGINE (TVAE — ثوانٍ)
# ─────────────────────────────────────────────

def generate_synthetic_tvae(df: pd.DataFrame, profile: dict, n_rows: int, epochs: int = 30, train_sample_size: int = 2000, progress_callback=None) -> pd.DataFrame:
    """
    يولّد DataFrame اصطناعي باستخدام TVAE (Tabular Variational Autoencoder).
    يتميز بأنه أسرع بكثير من CTGAN ويحافظ على دقة ممتازة.
    """
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import Metadata

    # حذف الأعمدة الحساسة + IDs + الأنماط الفريدة مؤقتاً
    excluded_cols = [
        c for c, v in profile.items()
        if not c.startswith("__") and (
            v.get("type", "").startswith("sensitive") or 
            v.get("type") == "id_sequence" or
            v.get("type") == "pattern_sequence"
        )
    ]
    df_clean = df.drop(columns=excluded_cols, errors="ignore")

    # ── عينة ذكية لتدريب TVAE ──
    n_train = len(df_clean)
    train_df = df_clean
    sampled_for_training = False
    if train_sample_size and n_train > train_sample_size:
        train_df = df_clean.sample(n=train_sample_size, random_state=42)
        sampled_for_training = True

    if progress_callback:
        msg = (
            f"جاري تحليل البيانات وبناء الـ Metadata..."
            if not sampled_for_training else
            f"✅ تحليل {n_train:,} صف — تدريب TVAE على عينة {train_sample_size:,} صف..."
        )
        progress_callback(0.1, msg)

    metadata = Metadata.detect_from_dataframe(data=train_df)
    metadata = _optimize_metadata_for_sdv(metadata, train_df, profile)

    if progress_callback:
        progress_callback(0.2, f"جاري تدريب نموذج TVAE... ({epochs} epoch، عينة {len(train_df):,} صف)")

    synthesizer = TVAESynthesizer(metadata, epochs=epochs, verbose=False)
    synthesizer.fit(train_df)

    if progress_callback:
        progress_callback(0.7, "جاري توليد البيانات الاصطناعية (TVAESampler)...")

    synthetic_df = synthesizer.sample(num_rows=n_rows)

    if progress_callback:
        progress_callback(0.85, "جاري إضافة البيانات الحساسة الوهمية والأعمدة الفريدة...")

    for col in excluded_cols:
        col_type = profile[col]["type"]
        if col_type == "id_sequence":
            start_val  = profile[col].get("min", 1)
            orig_prefix = profile[col].get("prefix", "")
            synthetic_df[col] = [f"SOL-{orig_prefix}{i}" for i in range(start_val, start_val + n_rows)]
        elif col_type == "pattern_sequence":
            template = profile[col]["template"]
            generated_vals = set()
            attempts = 0
            required_unique = n_rows
            while len(generated_vals) < required_unique and attempts < required_unique * 3:
                generated_vals.add(fake.bothify(template))
                attempts += 1
            val_list = list(generated_vals)
            if len(val_list) < n_rows:
                val_list += [fake.bothify(template) for _ in range(n_rows - len(val_list))]
            synthetic_df[col] = val_list[:n_rows]
        else:
            synthetic_df[col] = _generate_sensitive(profile[col], n_rows)

    synthetic_df = synthetic_df[df.columns]

    if progress_callback:
        progress_callback(1.0, "تم الانتهاء!")

    return synthetic_df


# ─────────────────────────────────────────────
# 2C. FAST AI ENGINE (GaussianCopula — ثوانٍ)
# ─────────────────────────────────────────────

def generate_synthetic_fast(df: pd.DataFrame, profile: dict, n_rows: int, progress_callback=None) -> pd.DataFrame:
    """
    يولد بيانات اصطناعية باستخدام GaussianCopulaSynthesizer — ثوانٍ وليس دقائق.
    الفرق عن CTGAN:
    - يعتمد على نموذج إحصائي (ليس شبكة عصبية) → لا epochs ولا انتظار.
    - يحاكي العلاقات بين الأعمدة بدقة عبر Gaussian Copulas.
    - مثالي للملفات الكبيرة أو عندما تحتاج نتيجة سريعة.
    """
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata

    if progress_callback:
        progress_callback(0.1, "⚙️ جاري تجهيز النموذج الإحصائي...")

    # حذف الأعمدة الحساسة + IDs + الأنماط الفريدة مؤقتاً
    excluded_cols = [
        c for c, v in profile.items()
        if not c.startswith("__") and (
            v.get("type", "").startswith("sensitive") or 
            v.get("type") == "id_sequence" or
            v.get("type") == "pattern_sequence"
        )
    ]
    df_clean = df.drop(columns=excluded_cols, errors="ignore")

    if progress_callback:
        progress_callback(0.2, "⚡ GaussianCopula يتدرّب (ثوانٍ)...")

    metadata = Metadata.detect_from_dataframe(data=df_clean)
    metadata = _optimize_metadata_for_sdv(metadata, df_clean, profile)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_clean)

    if progress_callback:
        progress_callback(0.7, "🎲 جاري توليد البيانات...")

    synthetic_df = synthesizer.sample(num_rows=n_rows)

    if progress_callback:
        progress_callback(0.85, "👤 إضافة البيانات الحساسة الوهمية والأعمدة الفريدة...")

    for col in excluded_cols:
        col_type = profile[col]["type"]
        if col_type == "id_sequence":
            start_val   = profile[col].get("min", 1)
            orig_prefix = profile[col].get("prefix", "")
            synthetic_df[col] = [f"SOL-{orig_prefix}{i}" for i in range(start_val, start_val + n_rows)]
        elif col_type == "pattern_sequence":
            template = profile[col]["template"]
            generated_vals = set()
            attempts = 0
            required_unique = n_rows
            while len(generated_vals) < required_unique and attempts < required_unique * 3:
                generated_vals.add(fake.bothify(template))
                attempts += 1
            val_list = list(generated_vals)
            if len(val_list) < n_rows:
                val_list += [fake.bothify(template) for _ in range(n_rows - len(val_list))]
            synthetic_df[col] = val_list[:n_rows]
        else:
            synthetic_df[col] = _generate_sensitive(profile[col], n_rows)

    synthetic_df = synthetic_df[df.columns]

    if progress_callback:
        progress_callback(1.0, "✅ تم الانتهاء!")

    return synthetic_df


# ─────────────────────────────────────────────
# 2C. NOISE & ANOMALY INJECTION (مختبر الضوضاء)
# ─────────────────────────────────────────────

def inject_noise(df: pd.DataFrame, profile: dict, null_pct: float = 0.0, outlier_pct: float = 0.0) -> pd.DataFrame:
    """
    يقوم بحقن ضوضاء وقيم شاذة في البيانات المولدة بقصد تدريب النماذج على بيانات غير مثالية.
    null_pct: نسبة القيم المفقودة المراد إضافتها عشوائياً.
    outlier_pct: نسبة القيم الشاذة (Outliers) المراد حقنها في الأعمدة الرقمية.
    """
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
    """
    يحسب نسبة التشابه لعمود واحد (0 → 100%).
    - الرقمي:   KS Test
    - الفئوي:  مقارنة توزيع التكرار
    - الحساس:  دايماً 100% (بيانات مولدة مستقلة)
    """
    if col_type == "sensitive":
        return 100.0

    if col_type == "numerical":
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
    """
    يرجع DataFrame يوضح Fidelity Score لكل عمود + المتوسط الكلي.
    """
    rows = []
    for col in original.columns:
        # تخطي الأعمدة الغير موجودة في synthetic (مثلاً في حالة الاستبعاد)
        if col not in synthetic.columns:
            continue
        if col not in profile or not isinstance(profile.get(col), dict):
            continue
        col_type = profile[col]["type"]
        score    = fidelity_score_column(original[col], synthetic[col], col_type)
        rows.append({
            "العمود":        col,
            "النوع":         col_type,
            "Fidelity Score (%)": score,
            "الجودة":        "ممتاز 🟢" if score >= 85 else "مقبول 🟡" if score >= 60 else "ضعيف 🔴"
        })

    report_df = pd.DataFrame(rows)

    # الصف الأخير: المتوسط الكلي
    avg_score = report_df["Fidelity Score (%)"].mean()
    total_row = pd.DataFrame([{
        "العمود":        "── المتوسط الكلي ──",
        "النوع":         "─",
        "Fidelity Score (%)": round(avg_score, 2),
        "الجودة":        "ممتاز 🟢" if avg_score >= 85 else "مقبول 🟡" if avg_score >= 60 else "ضعيف 🔴"
    }])

    return pd.concat([report_df, total_row], ignore_index=True)


# ─────────────────────────────────────────────
# 4. PRIVACY ANALYSIS  (تحليل الخصوصية)
# ─────────────────────────────────────────────

def _prepare_numeric_matrix(df: pd.DataFrame, profile: dict) -> np.ndarray:
    """
    يحول DataFrame إلى مصفوفة رقمية للمقارنة.
    - الأعمدة الرقمية: يتم تطبيعها (Normalize) بين 0 و 1.
    - الأعمدة الفئوية: يتم تحويلها إلى أرقام (Label Encoding).
    - الأعمدة الحساسة: يتم تجاهلها (لأنها مولدة بـ Faker ولا معنى لمقارنتها).
    """
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
    """
    يحسب Distance to Closest Record (DCR) لكل صف اصطناعي.
    """
    from scipy.spatial.distance import cdist

    orig_matrix = _prepare_numeric_matrix(original, profile)
    synth_matrix = _prepare_numeric_matrix(synthetic, profile)

    if orig_matrix.size == 0 or synth_matrix.size == 0:
        return {
            "dcr_values": np.array([]),
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


def generate_privacy_report(original: pd.DataFrame, synthetic: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """
    يولد تقرير خصوصية مفصل بجدول يمكن عرضه في الواجهة.
    """
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
    """يولد نص Markdown يحتوي على قاموس مفصل لجميع الأعمدة في البيانات."""
    md = "# 📚 قاموس البيانات (Data Dictionary)\n\n"
    md += f"**إجمالي الصفوف:** {len(df)}\n"
    md += f"**إجمالي الأعمدة:** {len(df.columns)}\n\n"
    md += "---\n\n"
    for col, info in profile.items():
        if col.startswith("__"): continue
        md += f"### 🔹 عمود: `{col}`\n"
        md += f"- **النوع (Type):** `{info.get('type', 'Unknown')}`\n"
        md += f"- **نسبة القيم المفقودة (Nulls):** {info.get('null_pct', 0):.2f}%\n"
        if info.get('type') == 'numerical':
            md += f"- **الحد الأدنى (Min):** {info.get('min', 'N/A')}\n"
            md += f"- **الحد الأقصى (Max):** {info.get('max', 'N/A')}\n"
            md += f"- **المتوسط (Mean):** {info.get('mean', 'N/A'):.2f}\n"
        elif info.get('type') == 'categorical':
            val_counts = info.get('value_counts', {})
            top_vals = list(val_counts.keys())[:5]
            md += f"- **أبرز الفئات المتكررة:** {', '.join([str(x) for x in top_vals])}\n"
        md += "\n"
    return md