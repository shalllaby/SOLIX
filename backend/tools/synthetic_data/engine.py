"""
SOL Platform — Synthetic Data Generation Engine
الموديول الأساسي: تحليل + توليد + تقرير
يدعم طريقتين: الإحصائية (Basic) والذكاء الاصطناعي (CTGAN, TVAE, GaussianCopula)
"""

import pandas as pd
import numpy as np
import random

class FallbackFaker:
    def __init__(self, locale="en_US"):
        self.locale = locale or "en_US"

    def name(self) -> str:
        import random
        if self.locale.startswith("ar"):
            first_names = ["أحمد", "محمد", "علي", "عمر", "فاطمة", "سارة", "ياسمين", "ليلى", "يوسف", "خالد", "عبدالله", "زينب", "مريم", "نور", "طارق", "منى"]
            last_names = ["الشافعي", "المصري", "الغانم", "العتيبي", "سليمان", "محمود", "حسن", "الخالدي", "الحربي", "صالح", "رضوان", "الرشيد"]
        else:
            first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Sarah", "Michael", "Amir", "Fatima", "Omar", "Layla", "Youssef", "Mariam"]
            last_names = ["Smith", "Doe", "Johnson", "Brown", "Taylor", "Miller", "Wilson", "Davis", "Anderson", "Thomas", "Jackson", "Al-Farsi", "Hassan", "Kamel", "Mansour"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
        
    def email(self) -> str:
        import random
        domains = ["example.com", "solix-agent.ai", "datafactory.io", "gmail.com", "yahoo.com", "outlook.com"]
        username = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
        return f"{username}@{random.choice(domains)}"
        
    def phone_number(self) -> str:
        import random
        if self.locale.startswith("ar"):
            return f"+966 5{random.randint(0,9)}{random.randint(1000000, 9999999)}"
        return f"+1-{random.randint(200, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        
    def address(self) -> str:
        import random
        if self.locale.startswith("ar"):
            streets = ["شارع الملك عبد العزيز", "شارع التخصصي", "طريق الملك فهد", "شارع العليا", "شارع جامعة الدول العربية"]
            cities = ["الرياض", "جدة", "الدمام", "مكة المكرمة", "القاهرة", "الإسكندرية", "دبي"]
            return f"{random.choice(streets)}، {random.choice(cities)}، {random.randint(10000, 99999)}"
        streets = ["Main St", "Oak Ave", "Pine Rd", "Maple Dr", "Broadway", "Science Park"]
        cities = ["Neo-City", "Cyber-Metropolis", "Data-Haven", "New York", "San Francisco", "London", "Cairo", "Dubai"]
        states = ["CA", "NY", "TX", "FL", "IL", "WA"]
        return f"{random.randint(100, 9999)} {random.choice(streets)}, {random.choice(cities)}, {random.choice(states)} {random.randint(10000, 99999)}"
        
    def ssn(self) -> str:
        import random
        return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        
    def bothify(self, text: str) -> str:
        import random
        result = []
        for char in text:
            if char == '#':
                result.append(str(random.randint(0, 9)))
            elif char == '?':
                result.append(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            else:
                result.append(char)
        return "".join(result)

    def date(self) -> str:
        import random
        return f"{random.randint(2015, 2026)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

    def date_between(self, start_date='-30y', end_date='today'):
        import random
        import datetime
        year = random.randint(2015, 2026)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return datetime.date(year, month, day)

    def random_element(self, elements):
        import random
        if isinstance(elements, dict):
            return random.choice(list(elements.keys()))
        return random.choice(list(elements))

    def sentence(self, nb_words=6) -> str:
        import random
        if self.locale.startswith("ar"):
            words = ["بيانات", "نظام", "تقرير", "عميل", "طلب", "حالة", "شحن", "توصيل", "دعم", "فني", "مراجعة", "مالي", "تحديث", "نجاح", "متابعة", "فاتورة", "تسليم", "مستندات", "معايير", "نشط", "حساب", "ملف", "تأكيد", "موافقة", "معالجة", "خطوة", "فريق", "عمليات", "سائق", "موقع"]
        else:
            words = ["system", "updated", "successfully", "transaction", "log", "customer", "notes", "delivery", "preferences", "support", "ticket", "created", "account", "manager", "annual", "sales", "report", "shows", "positive", "growth", "payment", "processed", "invoice", "generated", "internal", "audit", "scheduled", "business", "week", "profile", "verified", "approved", "compliance", "required", "documentation", "id", "verification", "address", "proof", "uploaded", "stored", "securely", "client", "premium", "tier", "cooldown", "period"]
        
        count = max(3, nb_words)
        sentence_words = [random.choice(words) for _ in range(count)]
        if sentence_words:
            sentence_words[0] = sentence_words[0].capitalize()
        return " ".join(sentence_words) + "."

    def text(self, max_nb_chars=200) -> str:
        import random
        sentences = [self.sentence(nb_words=random.randint(5, 12)) for _ in range(random.randint(2, 4))]
        text_str = " ".join(sentences)
        return text_str[:max_nb_chars]

    def company(self) -> str:
        import random
        if self.locale.startswith("ar"):
            names = ["شركة الرياض للتجارة", "مؤسسة الحلول الرقمية", "مجموعة الفوزان", "شركة التقنية المتقدمة"]
            return random.choice(names)
        names = ["Tech Corp", "Data Dynamics", "Global Solutions", "Future Industries"]
        return random.choice(names)

    def job(self) -> str:
        import random
        if self.locale.startswith("ar"):
            jobs = ["مهندس برمجيات", "محاسب مالي", "مدير مشاريع", "محلل بيانات", "طبيب استشاري", "معلم مدرسة"]
            return random.choice(jobs)
        jobs = ["Software Engineer", "Financial Analyst", "Project Manager", "Data Analyst", "Consultant", "Teacher"]
        return random.choice(jobs)

    def city(self) -> str:
        import random
        if self.locale.startswith("ar"):
            return random.choice(["الرياض", "جدة", "الدمام", "مكة المكرمة", "المدينة المنورة", "القاهرة", "دبي"])
        return random.choice(["New York", "San Francisco", "London", "Cairo", "Dubai", "Riyadh"])

    def country(self) -> str:
        import random
        if self.locale.startswith("ar"):
            return random.choice(["المملكة العربية السعودية", "مصر", "الإمارات العربية المتحدة", "الأردن"])
        return random.choice(["United States", "United Kingdom", "Saudi Arabia", "Egypt", "United Arab Emirates"])

    def __getattr__(self, name):
        import random
        name_lower = name.lower()
        if 'paragraph' in name_lower:
            return lambda *args, **kwargs: self.text()
        if 'word' in name_lower:
            return lambda *args, **kwargs: random.choice(["data", "system", "value", "report", "process", "standard"])
        if 'first_name' in name_lower:
            return lambda *args, **kwargs: self.name().split()[0]
        if 'last_name' in name_lower:
            return lambda *args, **kwargs: self.name().split()[-1]
        return lambda *args, **kwargs: self.sentence()

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = FallbackFaker()

from scipy.stats import ks_2samp
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import os
warnings.filterwarnings("ignore")

# 🔑 المفتاح البديل لتشغيل ذكاء الفهم العميق في حال عدم إعداده في البيئة
BACKUP_GROQ_API_KEY = None

# ─── رسالة الخطأ المناسبة لعدم توفر مكتبة sdv ───
IS_KAGGLE = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if IS_KAGGLE:
    SDV_IMPORT_ERROR = (
        "مكتبة sdv غير مثبتة على Kaggle Cloud. "
        "يرجى تفعيل خيار الإنترنت (Internet) في إعدادات حساب Kaggle الخاص بك "
        "(عبر التحقق من رقم الهاتف وتفعيل خيار 'Internet' في لوحة الإعدادات الجانبية للنوت بوك) "
        "لتتمكن البيئة من تثبيت مكتبة sdv تلقائياً لتشغيل النماذج المتقدمة."
    )
else:
    SDV_IMPORT_ERROR = "مكتبة sdv غير مثبتة محلياً. يرجى تثبيتها باستخدام 'pip install sdv' أو تشغيل التوليد عبر Kaggle Cloud."

# ─── Cache داخلي لنتائج LLM (يمنع إعادة الاستدعاء للملف نفسه) ───
_LLM_CACHE: dict = {}


# ─────────────────────────────────────────────
# 1. AI-DRIVEN STATISTICAL PROFILING  (تحليل الهيكل بالفهم العميق)
# ─────────────────────────────────────────────

def _col_signature(df: pd.DataFrame) -> str:
    """بصمة فريدة للأعمدة تُسخدم كمفتاح Cache — سريعة جداً."""
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


def detect_column_types_llm(df: pd.DataFrame, api_key: Optional[str] = None, user_id: Optional[int] = None) -> dict:
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
            if user_id:
                from backend.store import active_user_id
                active_user_id.set(user_id)
            from groq import Groq
            key = api_key or os.getenv("GROQ_API_KEY", "").strip()
            if not key or not key.startswith("gsk"):
                raise ValueError("No valid API key")
            client = Groq(api_key=key)
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                try:
                    from backend.utils.llm_logger import log_groq_response
                    log_groq_response(resp, module_name="synthetic")
                except:
                    pass
                result_container["result"] = json.loads(resp.choices[0].message.content)
            except Exception as first_err:
                err_msg = str(first_err)
                if "429" in err_msg or "rate limit" in err_msg.lower() or "limit exceeded" in err_msg.lower() or "too many requests" in err_msg.lower():
                    # Output log exactly as requested
                    print('INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"')
                    backup_model = "meta-llama/llama-4-scout-17b-16e-instruct"
                    print(f"Switching automatically to backup model: {backup_model}")
                    resp = client.chat.completions.create(
                        model=backup_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    )
                    try:
                        from backend.utils.llm_logger import log_groq_response
                        log_groq_response(resp, module_name="synthetic")
                    except:
                        pass
                    result_container["result"] = json.loads(resp.choices[0].message.content)
                else:
                    raise first_err
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
            fallback = _fallback_types(df)
            for c in df.columns:
                if c not in raw_result:
                    raw_result[c] = fallback.get(c, "categorical")
    else:
        # لو حدث خطأ أو انتهت الـ 10 ثوانٍ → الانتقال فوراً للـ Fallback
        raw_result = _fallback_types(df)

    # حفظ النتيجة في الكاش
    _LLM_CACHE[cache_key] = raw_result
    return raw_result


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

def profile_dataframe(df: pd.DataFrame, user_id: Optional[int] = None) -> dict:
    """
    يدرس كل عمود ويرجع قاموس بخصائصه الإحصائية مدمجاً بذكاء LLM للفهم.
    """
    import re

    profile = {}

    profile["__meta__"] = {
        "total_rows":    len(df),
        "analyzed_rows": len(df),
        "was_sampled":   False,
    }

    llm_types = detect_column_types_llm(df, user_id=user_id)

    null_pcts   = df.isnull().mean() * 100
    num_uniques = df.nunique()

    for col in df.columns:
        col_type = llm_types.get(col, "categorical")

        if col_type == "numerical" and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                col_type = "categorical"

        if col_type.startswith("sensitive") and pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numerical"

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
                        nums = df[col].dropna().astype(str).str.replace(pot_prefix, '', regex=False).astype(int)
                        start_num = int(nums.min())

        if is_id:
            col_type = "id_sequence"

        template = ""
        if col_type == "categorical":
            if n_unique > 20 and (n_unique / n_total) > 0.05:
                template = extract_pattern_template(df[col])
                if template and ("#" in template or "?" in template):
                    col_type = "pattern_sequence"

        info = {"type": col_type, "null_pct": null_pcts[col]}

        if col_type == "numerical":
            info.update({
                "mean":   float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0,
                "std":    float(df[col].std()) if not pd.isna(df[col].std()) else 0.0,
                "min":    float(df[col].min()) if not pd.isna(df[col].min()) else 0.0,
                "max":    float(df[col].max()) if not pd.isna(df[col].max()) else 0.0,
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

    num_cols = [c for c, v in profile.items() if isinstance(v, dict) and v.get("type") == "numerical"]
    if len(num_cols) > 1:
        profile["__correlations__"] = df[num_cols].corr().fillna(0).to_dict()

    return profile


# ─────────────────────────────────────────────
# 2A. BASIC GENERATION ENGINE  (توليد إحصائي — مع Threads)
# ─────────────────────────────────────────────

def _generate_numerical(info: dict, n: int) -> pd.Series:
    values = np.random.normal(loc=info["mean"], scale=info["std"] if info["std"] > 0 else 1, size=n)
    values = np.clip(values, info["min"], info["max"])
    if info.get("is_int"):
        values = np.round(values).astype(int)
    return pd.Series(values)


def _generate_categorical(info: dict, n: int) -> pd.Series:
    categories = list(info["value_counts"].keys())
    weights    = list(info["value_counts"].values())
    # تطبيع الأوزان لتجنب خطأ المجموع
    weights = np.array(weights) / np.sum(weights)
    return pd.Series(np.random.choice(categories, size=n, p=weights))


def _generate_sensitive(info: dict, n: int) -> pd.Series:
    col_type = info["type"]
    if col_type == "sensitive_email":
        values = [fake.email() for _ in range(n)]
    elif col_type == "sensitive_phone":
        values = [fake.phone_number() for _ in range(n)]
    elif col_type == "sensitive_address":
        values = [fake.address().replace("\n", ", ") for _ in range(n)]
    elif col_type == "sensitive_id":
        values = [fake.ssn() for _ in range(n)]
    else:
        values = [fake.name() for _ in range(n)]
    return pd.Series(values)


def _generate_single_column(args):
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
    """
    tasks = [(col, profile[col], n_rows) for col in df.columns if col in profile]

    synthetic = {}

    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
        futures = {executor.submit(_generate_single_column, task): task[0] for task in tasks}
        for future in as_completed(futures):
            col, series = future.result()
            synthetic[col] = series

    result = pd.DataFrame({col: synthetic[col] for col in df.columns if col in synthetic})

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
    يولّد DataFrame اصطناعي باستخدام CTGAN.
    """
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import Metadata
    except ImportError:
        if progress_callback:
            progress_callback(0.1, "⚠️ تنبيه: نموذج CTGAN يتطلب مكتبة sdv وهي غير مثبتة (بسبب عدم وجود إنترنت على كاجل).")
            progress_callback(0.2, "⚙️ جاري التحويل تلقائياً إلى نموذج Gaussian Copula المدمج لإنقاذ العملية وتوليد البيانات...")
        return generate_synthetic_fast(df, profile, n_rows, progress_callback=progress_callback)

    excluded_cols = [
        c for c, v in profile.items()
        if not c.startswith("__") and (
            v.get("type", "").startswith("sensitive") or 
            v.get("type") == "id_sequence" or
            v.get("type") == "pattern_sequence"
        )
    ]
    df_clean = df.drop(columns=excluded_cols, errors="ignore")

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
    try:
        synthesizer.fit(train_df)
    except Exception as e:
        err_msg = str(e)
        if "CUDA" in err_msg or "cuda" in err_msg or "Accelerator" in err_msg:
            if progress_callback:
                progress_callback(0.25, "⚠️ تعذر تدريب نموذج CTGAN بسبب مشكلة توافق كارت الشاشة (CUDA Error) في كاجل.")
                progress_callback(0.3, "⚙️ جاري التحويل التلقائي لنموذج Gaussian Copula المدمج لإنقاذ العملية وتوليد البيانات...")
            return generate_synthetic_fast(df, profile, n_rows, progress_callback=progress_callback)
        else:
            raise e

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


# ─── 2B_2. FAST AI ENGINE (TVAE — ثوانٍ) ───

def generate_synthetic_tvae(df: pd.DataFrame, profile: dict, n_rows: int, epochs: int = 30, train_sample_size: int = 2000, progress_callback=None) -> pd.DataFrame:
    """
    يولّد DataFrame اصطناعي باستخدام TVAE.
    """
    try:
        from sdv.single_table import TVAESynthesizer
        from sdv.metadata import Metadata
    except ImportError:
        if progress_callback:
            progress_callback(0.1, "⚠️ تنبيه: نموذج TVAE يتطلب مكتبة sdv وهي غير مثبتة (بسبب عدم وجود إنترنت على كاجل).")
            progress_callback(0.2, "⚙️ جاري التحويل تلقائياً إلى نموذج Gaussian Copula المدمج لإنقاذ العملية وتوليد البيانات...")
        return generate_synthetic_fast(df, profile, n_rows, progress_callback=progress_callback)

    excluded_cols = [
        c for c, v in profile.items()
        if not c.startswith("__") and (
            v.get("type", "").startswith("sensitive") or 
            v.get("type") == "id_sequence" or
            v.get("type") == "pattern_sequence"
        )
    ]
    df_clean = df.drop(columns=excluded_cols, errors="ignore")

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
    try:
        synthesizer.fit(train_df)
    except Exception as e:
        err_msg = str(e)
        if "CUDA" in err_msg or "cuda" in err_msg or "Accelerator" in err_msg:
            if progress_callback:
                progress_callback(0.25, "⚠️ تعذر تدريب نموذج TVAE بسبب مشكلة توافق كارت الشاشة (CUDA Error) في كاجل.")
                progress_callback(0.3, "⚙️ جاري التحويل التلقائي لنموذج Gaussian Copula المدمج لإنقاذ العملية وتوليد البيانات...")
            return generate_synthetic_fast(df, profile, n_rows, progress_callback=progress_callback)
        else:
            raise e

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


class PureGaussianCopula:
    def __init__(self):
        self.mappings = {}
        self.columns = []
        self.cov_matrix = None
        self.means = None
        self.numerical_cols = []
        self.categorical_cols = []
        
    def fit(self, df: pd.DataFrame, profile: dict):
        from scipy.stats import norm
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
        cov = np.cov(Z.T)
        self.cov_matrix = cov + np.eye(len(self.columns)) * 1e-6
        
    def sample(self, num_rows: int) -> pd.DataFrame:
        from scipy.stats import norm
        if len(self.columns) == 1:
            cov_val = float(self.cov_matrix) if np.isscalar(self.cov_matrix) or self.cov_matrix.ndim == 0 else float(self.cov_matrix[0, 0])
            z_sampled = np.random.normal(self.means[0], np.sqrt(max(1e-6, cov_val)), size=(num_rows, 1))
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


# ─────────────────────────────────────────────
# 2C. FAST AI ENGINE (GaussianCopula — ثوانٍ)
# ─────────────────────────────────────────────

def generate_synthetic_fast(df: pd.DataFrame, profile: dict, n_rows: int, progress_callback=None) -> pd.DataFrame:
    """
    يولد بيانات اصطناعية باستخدام GaussianCopulaSynthesizer.
    """
    use_pure_copula = False
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import Metadata
    except ImportError:
        use_pure_copula = True

    excluded_cols = [
        c for c, v in profile.items()
        if not c.startswith("__") and (
            v.get("type", "").startswith("sensitive") or 
            v.get("type") == "id_sequence" or
            v.get("type") == "pattern_sequence"
        )
    ]
    df_clean = df.drop(columns=excluded_cols, errors="ignore")

    if use_pure_copula:
        if progress_callback:
            progress_callback(0.15, "⚙️ مكتبة sdv غير مثبتة. جاري تشغيل نموذج Gaussian Copula المدمج ذو الكفاءة العالية...")
            progress_callback(0.35, "⚡ جاري تدريب نموذج Gaussian Copula المدمج...")
        
        copula = PureGaussianCopula()
        copula.fit(df_clean, profile)
        
        if progress_callback:
            progress_callback(0.75, "🎲 جاري توليد البيانات من النموذج المدمج...")
            
        synthetic_df = copula.sample(n_rows)
    else:
        if progress_callback:
            progress_callback(0.1, "⚙️ جاري تجهيز النموذج الإحصائي...")

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
            "dcr_values": [],
            "mean_dcr": 0.0,
            "min_dcr": 0.0,
            "max_dcr": 0.0,
            "privacy_score": 0.0,
            "risk_level": "غير قابل للحساب",
            "at_risk_count": 0,
            "at_risk_pct": 0.0,
            "risk_threshold": 0.0,
            "total_checked": 0
        }

    if len(synth_matrix) > sample_size:
        idx = np.random.choice(len(synth_matrix), sample_size, replace=False)
        synth_sample = synth_matrix[idx]
    else:
        synth_sample = synth_matrix

    distances = cdist(synth_sample, orig_matrix, metric='euclidean')
    dcr_values = distances.min(axis=1)

    # Convert any potential NaN / Inf values to clean defaults before rounding
    mean_dcr_val = np.mean(dcr_values)
    mean_dcr = float(mean_dcr_val) if not (np.isnan(mean_dcr_val) or np.isinf(mean_dcr_val)) else 0.0

    min_dcr_val = np.min(dcr_values)
    min_dcr = float(min_dcr_val) if not (np.isnan(min_dcr_val) or np.isinf(min_dcr_val)) else 0.0

    max_dcr_val = np.max(dcr_values)
    max_dcr = float(max_dcr_val) if not (np.isnan(max_dcr_val) or np.isinf(max_dcr_val)) else 0.0

    risk_threshold = mean_dcr * 0.15
    at_risk = (dcr_values < risk_threshold).sum()
    at_risk_pct = round((at_risk / len(dcr_values)) * 100, 2) if len(dcr_values) > 0 else 0.0
    
    if np.isnan(at_risk_pct) or np.isinf(at_risk_pct):
        at_risk_pct = 0.0
        
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
        "dcr_values": [float(x) for x in dcr_values.tolist()],
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


def suggest_schema_from_prompt(user_prompt: str, num_columns: int = 5, locale: str = "en_US", api_key: Optional[str] = None, user_id: Optional[int] = None) -> dict:
    """
    يقترح هيكل بيانات (Schema) وقواعد العلاقات المنطقية (Rules) بناءً على وصف المستخدم والـ locale المطلوبة.
    """
    import json
    import threading
    import os

    target_language = "Arabic" if locale.startswith("ar") else "English"

    prompt = f"""أنت خبير علم بيانات ومبرمج بايثون محترف للغاية.
المطلوب منك هو اقتراح هيكل بيانات (Schema) لجدول يحتوي على {num_columns} أعمدة بالإضافة إلى قائمة القواعد المنطقية والارتباطات (Rules) الممكنة بين الأعمدة لتطبيقها في التوليد، وكل ذلك يلائم وصف المستخدم التالي:
\"{user_prompt}\"

اللغة/الدولة المستهدفة (locale): {locale}

الأنواع المدعومة للأعمدة (type) التي يجب عليك الاختيار منها حصراً هي:
1. \"id\": رقم تسلسلي أو معرّف فريد.
2. \"name\": اسم شخص كامل.
3. \"email\": بريد إلكتروني.
4. \"phone\": رقم هاتف متوافق مع سياق الوصف والدولة.
5. \"address\": عنوان جغرافي.
6. \"number\": أرقام وحسابات (مثال: السعر، العمر، الكمية). حدد \"min\" و \"max\" و \"is_float\" في التفاصيل.
7. \"category\": فئات/تصنيفات خيارات محددة. أرفق قائمة بالخيارات المقترحة \"categories\" مثلاً: [\"رياضة\", \"ألعاب\", \"موسيقى\"].
8. \"date\": تاريخ (مثلاً تاريخ ميلاد، تاريخ شراء).
9. \"text\": نص حر قصير (مثلاً ملاحظات، وصف منتج).

القواعد المنطقية والرياضية المدعومة التي يجب اقتراحها لربط الأعمدة ببعضها (rules) هي:
1. قواعد المعادلات الرياضية (formula): لحساب قيمة عمود بناءً على عملية حسابية أو جبرية من أعمدة أخرى.
   مثال: {{"type": "formula", "target": "total_price", "expression": "price * quantity + tax"}}
2. القواعد الشرطية (conditional): لوضع شروط على الأعمدة لتعديل قيمها أو فرض منطق معين.
   أدوات المقارنة المتاحة: "==", "!=", ">", "<", ">=", "<=".
   مثال: {{"type": "conditional", "if_col": "age", "if_op": "<", "if_val": "18", "then_col": "ticket_price", "then_val": "ticket_price * 0.5"}}
3. قواعد الارتباط (correlation): لفرض ارتباط إيجابي أو سلبي قوي (معامل ارتباط > 0.6 أو < -0.6) بين عمودين رقميين.
   الجهات المتاحة للارتباط: "positive" أو "negative".
   مثال: {{"type": "correlation", "col1": "experience_years", "col2": "salary", "direction": "positive"}}

صيغة الإرجاع يجب أن تكون كائن JSON فقط يحتوي على قائمة من الأعمدة وقائمة من القواعد بالصيغة التالية:
{{
  \"columns\": [
    {{
      \"name\": \"column_name_in_english\",
      \"display_name\": \"اسم العمود باللغة المطلوبة\",
      \"type\": \"type_name\",
      \"description\": \"وصف مختصر للعمود باللغة المطلوبة\",
      \"min\": 10,  // اختياري فقط إذا كان النوع number
      \"max\": 100, // اختياري فقط إذا كان النوع number
      \"is_float\": false, // اختياري فقط إذا كان النوع number
      \"categories\": [\"cat1\", \"cat2\"] // إجباري فقط إذا كان النوع category
    }}
  ],
  \"rules\": [
    // القواعد والارتباطات المقترحة لربط هذه الأعمدة ببعضها منطقياً ورياضياً
  ]
}}

تعليمات صارمة جداً بشأن تطابق اللغة والمحلية (Locale & Language Consistency):
- The user requested the locale '{locale}'. Therefore, the target language for this dataset is {target_language}.
- You MUST generate all values for 'display_name', 'description', and 'categories' strictly and entirely in {target_language}.
- NO LANGUAGE MIXING: If {target_language} is Arabic, under no circumstances should you return English words inside the 'categories', 'display_name', or 'description' fields.
- The ONLY exception is the programmatic 'name' field, which must always remain in English snake_case (e.g., 'car_color').
"""

    result_container = {"result": None, "error": None}

    def _call_llm():
        try:
            if user_id:
                from backend.store import active_user_id
                active_user_id.set(user_id)
            from groq import Groq
            key = api_key or os.getenv("GROQ_API_KEY", "").strip()
            if not key or not key.startswith("gsk"):
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
                try:
                    from backend.utils.llm_logger import log_groq_response
                    log_groq_response(resp, module_name="synthetic")
                except:
                    pass
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
                    try:
                        from backend.utils.llm_logger import log_groq_response
                        log_groq_response(resp, module_name="synthetic")
                    except:
                        pass
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
    rules = []
    if result_container["result"] is not None:
        columns = result_container["result"].get("columns", [])
        rules = result_container["result"].get("rules", [])
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

    # تعقيم وتأكيد تطابق الأنواع المدعومة للأعمدة
    allowed_types = {"id", "name", "email", "phone", "address", "number", "category", "date", "text"}
    sanitized_columns = []
    for col in columns:
        col_type = col.get("type", "text").lower()
        if col_type not in allowed_types:
            col_type = "text"
        col["type"] = col_type
        sanitized_columns.append(col)

    # تعقيم القواعد المسترجعة والتأكد من مطابقتها
    sanitized_rules = []
    for r in rules:
        if not isinstance(r, dict) or "type" not in r:
            continue
        rtype = r["type"]
        if rtype == "formula" and "target" in r and "expression" in r:
            sanitized_rules.append({
                "type": "formula",
                "target": r["target"],
                "expression": r["expression"]
            })
        elif rtype == "conditional" and "if_col" in r and "if_op" in r and "if_val" in r and "then_col" in r and "then_val" in r:
            sanitized_rules.append({
                "type": "conditional",
                "if_col": r["if_col"],
                "if_op": r["if_op"],
                "if_val": r["if_val"],
                "then_col": r["then_col"],
                "then_val": r["then_val"]
            })
        elif rtype == "correlation" and "col1" in r and "col2" in r:
            sanitized_rules.append({
                "type": "correlation",
                "col1": r["col1"],
                "col2": r["col2"],
                "direction": r.get("direction", "positive")
            })

    return {"columns": sanitized_columns, "rules": sanitized_rules}


def generate_data_via_code_agent(user_prompt: str, schema_columns: list, rules: list, num_rows: int, locale: str = "en_US", model: str = "llama-3.3-70b-versatile", user_id: int = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    يستخدم Groq LLM لكتابة كود بايثون مخصص لتوليد البيانات مع الحفاظ على العلاقات والارتباطات الإحصائية.
    """
    import threading
    import json
    import os
    import re

    # تجهيز توصيف القواعد لحقنها في البرومبت
    rules_instruction = ""
    if rules:
        rules_instruction = "\nقواعد الربط والاعتمادات التي يجب تطبيقها وتشفيرها بدقة صرامة تامة في الكود:\n"
        for i, rule in enumerate(rules):
            rtype = rule.get("type")
            if rtype == "formula":
                rules_instruction += f"   - قاعدة معادلة رياضية (Formula): العمود '{rule.get('target')}' يجب حسابه باستخدام التعبير الرياضي/الجبري التالي: {rule.get('expression')}\n"
            elif rtype == "conditional":
                rules_instruction += f"   - قاعدة شرط منطقي (Conditional): إذا كان العمود '{rule.get('if_col')}' {rule.get('if_op')} {rule.get('if_val')}، فيجب أن يعين للعمود '{rule.get('then_col')}' القيمة/التعبير: {rule.get('then_val')}\n"
            elif rtype == "correlation":
                rules_instruction += f"   - قاعدة ارتباط إحصائي (Correlation): يجب فرض ارتباط إحصائي {rule.get('direction')} قوي (معامل ارتباط أكبر من 0.6 أو أقل من -0.6) بين العمودين '{rule.get('col1')}' و '{rule.get('col2')}'\n"

    # تجهيز البرومبت
    prompt = f"""أنت خبير علم بيانات ومبرمج بايثون محترف للغاية.
المطلوب منك هو كتابة كود بايثون لتوليد بيانات اصطناعية تحاكي وصف المستخدم التالي:
"{user_prompt}"

الهيكل المختار للجدول (Schema) هو:
{json.dumps(schema_columns, indent=2, ensure_ascii=False)}

اللغة/الدولة المستهدفة (locale): {locale}

يجب عليك كتابة دالة بايثون وحيدة بالاسم والتوقيع التالي:
`def generate_data(num_rows: int) -> pd.DataFrame:`

تعليمات هامة جداً وقواعد صارمة لا غنى عنها لضمان دقة البيانات للتعلم الآلي (Machine Learning):
1. شمولية الكود لجميع الأعمدة:
   يجب توليد كود لجميع الأعمدة المحددة في الهيكل (schema) بالكامل دون استثناء أي عمود (إجمالي الأعمدة المطلوب توليدها: {len(schema_columns)} عمود). لا تستخدم الاختصارات أو التوقف التلقائي أو تعليقات مثل '# الباقي هنا...' أو '# وهكذا...'. يجب كتابة الكود كاملاً لجميع الأعمدة الـ {len(schema_columns)} دون إغفال أي عمود.
2. معالجة الأعمدة ذات التصنيفات المحددة (categories):
   إذا كان العمود يحتوي على قائمة "categories" محددة في الهيكل (schema), يجب عليك توليد القيم الخاصة به حصراً من تلك القائمة باستخدام `random.choices(categories, k=num_rows)` أو `np.random.choice` وعدم استخدام أي قيم خارجها.
3. بناء علاقات وارتباطات منطقية وإحصائية صارمة وجريئة (Strict Logical & Mathematical Correlations) باستخدام كود vectorized:
   {rules_instruction}
   - يجب دائماً بناء هذه العلاقات المتبادلة والارتباطات المفروضة أعلاه وبقية العلاقات المنطقية الضمنية السياقية باستخدام عمليات NumPy المتجهة (Vectorized) السريعة جداً (مثل `np.where` أو `np.select` أو عمليات الضرب والجمع الجبري).
   - قم أولاً بتوليد الأعمدة المستقلة (مثل سنة الصنع، نوع المتجر، المساحة، أو العمر).
   - قم بتوليد الأعمدة التابعة بناءً على الأعمدة المستقلة لتعكس القواعد المنطقية والمبرهنة (مثل تطبيق الشروط أو المعادلات الرياضية المذكورة في القواعد).
   - تأكد أن يكون معامل الارتباط (Correlation Coefficient) بين المتغيرات المرتبطة إيجابياً أو سلبياً مرتفعاً جداً (أكبر من 0.6 أو أقل من -0.6 حسب طبيعة العلاقة).
   - أضف تشويشاً إحصائياً خفيفاً وواقعياً (Statistical Noise) بعد تحديد العلاقة الأساسية باستخدام `np.random.normal(0, std, size=num_rows)` مع ضمان عدم كسر العلاقة المنطقية الأساسية والقواعد المحددة.
4. حماية القيم من التجاوز غير المنطقي (Safety Wrapping):
   عند إضافة التشويش الإحصائي (Noise) للأسعار أو الأرقام، تأكد من التفافها وحمايتها باستخدام `np.clip` لمنع حدوث قيم سالبة أو غير منطقية (مثلاً الأسعار أو الأعمار يجب ألا تقل عن 0 أو الحد الأدنى المسموح به).
5. التطابق التام مع الـ Locale واللغة المطلوبة واستخدم مزودي Faker المناسبين:
   - يمنع منعاً باتاً توليد جمل طويلة (مثل `fake.sentence()`) للأعمدة القياسية القصيرة مثل أسماء المتاجر/الشركات أو المدن.
   - إذا كان العمود يمثل اسماً (مثل `store_name` أو `company_name`) استخدم `fake.company()`, وإذا كان يمثل مدينة استخدم `fake.city()`.
   - لا تستخدم `fake.sentence()` أو `fake.paragraph()` إلا إذا كان العمود مخصصاً صراحة للنصوص الطويلة (مثل `notes` أو `description`).
   - إذا كانت الـ locale المطلوبة تبدأ بـ "ar" (مثل ar_EG)، فيجب كتابة كافة المخرجات النصية المخصصة والتصنيفات (Categorical values) باللغة العربية حصراً داخل الكود (مثال: ["جديد", "مستعمل"] بدلاً من ["New", "Used"]).
   - يمنع منعاً باتاً خلط نصوص إنجليزية في الخيارات أو الشروط (مثال: استخدام `np.where(years >= 2022, 'جديد', 'مستعمل')` وليس `np.where(years >= 2022, 'New', 'Used')`).
   - بالنسبة للأعمدة القياسية (مثل الاسم، العنوان، البريد، التاريخ)، استخدم الكائن `fake` الممرر (وهو مهيأ مسبقاً بالـ locale المطلوبة) لإنشاء قيم متوافقة ومناسبة (مثل `[fake.name() for _ in range(num_rows)]`).
6. يجب أن ترجع الدالة DataFrame يحتوي على عدد الصفوف المطلوب بالضبط (num_rows) وبأسماء الأعمدة المطابقة للهيكل تماماً.
7. كتابة كود بايثون حقيقي وقابل للتنفيذ بالكامل (No Pseudo-code):
   - يجب أن يكون الكود المولد كود بايثون حقيقي تماماً، خالٍ من الأخطاء النحوية (syntax errors)، ومستعداً للتشغيل الفوري داخل exec().
   - يمنع تماماً كتابة كود وهمي (pseudo-code)، أو استدعاء دوال أو متغيرات غير معرفة أو غير مستوردة، أو استخدام نقاط الحذف (مثل `...` أو `# الباقي هنا`).
   - تأكد من أن جميع الكائنات والمكتبات المستخدمة (مثل `import pandas as pd`, `import numpy as np`, `import random`) مستوردة ومعرفة بشكل صحيح.
8. أرجع كود البايثون فقط داخل وسم الكود البصري ```python ولا تكتب أي مقدمات أو شروحات إطلاقاً.
"""

    result_container = {"code": None, "error": None}

    def _call_llm():
        try:
            if user_id:
                from backend.store import active_user_id
                active_user_id.set(user_id)
            from groq import Groq
            key = api_key or os.getenv("GROQ_API_KEY", "").strip()
            if not key or not key.startswith("gsk"):
                raise ValueError("No valid API key")
            client = Groq(api_key=key)
            
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=4096,
                )
                try:
                    from backend.utils.llm_logger import log_groq_response
                    log_groq_response(resp, module_name="synthetic")
                except:
                    pass
                result_container["code"] = resp.choices[0].message.content
            except Exception as first_err:
                err_msg = str(first_err)
                if "429" in err_msg or "rate limit" in err_msg.lower() or "limit exceeded" in err_msg.lower() or "too many requests" in err_msg.lower():
                    # Output log exactly as requested
                    print('INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"')
                    backup_model = "meta-llama/llama-4-scout-17b-16e-instruct"
                    print(f"Switching automatically to backup model: {backup_model}")
                    
                    if user_id:
                        try:
                            from backend.database import SessionLocal
                            from backend.models import Notification
                            db = SessionLocal()
                            notification = Notification(
                                user_id=user_id,
                                title="تغيير تلقائي لنموذج الذكاء الاصطناعي (Groq 429)",
                                message=(
                                    f"تم الوصول إلى حد الاستخدام للنموذج الأساسي (429 Too Many Requests).\n"
                                    f"تم التبديل تلقائياً في نفس اللحظة إلى النموذج الاحتياطي ({backup_model}) وتكملة نفس العملية."
                                ),
                                type="info"
                            )
                            db.add(notification)
                            db.commit()
                            db.close()
                        except Exception as db_err:
                            print(f"Failed to log switch success notification: {db_err}")
                            
                    resp = client.chat.completions.create(
                        model=backup_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=4096,
                    )
                    try:
                        from backend.utils.llm_logger import log_groq_response
                        log_groq_response(resp, module_name="synthetic")
                    except:
                        pass
                    result_container["code"] = resp.choices[0].message.content
                else:
                    raise first_err
        except Exception as e:
            result_container["error"] = str(e)

    # تشغيل في Thread مع Timeout 30 ثانية
    t = threading.Thread(target=_call_llm, daemon=True)
    t.start()
    t.join(timeout=30)

    if result_container["error"]:
        raise ValueError(result_container["error"])
    if not result_container["code"]:
        raise ValueError("LLM execution timed out")

    # استخراج كود البايثون من الإجابة
    raw_content = result_container["code"]
    code_match = re.search(r"```python\s*(.*?)\s*```", raw_content, re.DOTALL)
    if code_match:
        python_code = code_match.group(1)
    else:
        python_code = raw_content.strip()

    # تهيئة الـ Faker لاستخدامه في الساندبوكس
    try:
        from faker import Faker
        fake_inst = Faker(locale)
        Faker_class = Faker
    except Exception:
        fake_inst = FallbackFaker(locale)
        Faker_class = FallbackFaker

    # Mock faker module in sys.modules if not installed to prevent ModuleNotFoundError inside sandbox
    import sys
    import types
    restored_faker = False
    original_faker = sys.modules.get("faker")
    if not original_faker or getattr(original_faker, "Faker", None) is None:
        mocked_faker = types.ModuleType("faker")
        mocked_faker.Faker = Faker_class
        sys.modules["faker"] = mocked_faker
        restored_faker = True

    import pandas as pd
    import numpy as np
    import random

    # تحضير ساندبوكس التشغيل
    sandbox = {
        "pd": pd,
        "np": np,
        "random": random,
        "fake": fake_inst,
        "Faker": Faker_class,
        "num_rows": num_rows,
        "locale": locale,
        "schema_columns": schema_columns,
        "__builtins__": globals()["__builtins__"],
    }

    try:
        # تنفيذ الكود
        exec(python_code, sandbox)

        if "generate_data" not in sandbox:
            raise ValueError("generate_data function was not defined in the generated code")

        generate_data_fn = sandbox["generate_data"]
        df = generate_data_fn(num_rows)

        if not isinstance(df, pd.DataFrame):
            raise ValueError("generate_data did not return a pandas DataFrame")

        print("Successfully applied logical correlations!")
        return df
    finally:
        if restored_faker:
            if original_faker is None:
                sys.modules.pop("faker", None)
            else:
                sys.modules["faker"] = original_faker


def apply_logical_rules(df: pd.DataFrame, rules: list, schema_columns: list = None) -> pd.DataFrame:
    """
    تطبيق القواعد والارتباطات المنطقية يدوياً على الـ DataFrame كإجراء أمان إضافي (Fallback)
    أو لضمان توافق البيانات التام مع رغبة المستخدم.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the generated synthetic data.
        rules (list): List of rule dicts (e.g. formula, conditional, correlation).
        schema_columns (list, optional): List of column definition dicts.
    """
    if df is None or df.empty or not rules:
        # If no rules but we still have data, perform phone number sanitization anyway
        if df is not None and not df.empty:
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample_series = df[col].dropna()
                    if not sample_series.empty and sample_series.astype(str).str.startswith('+').any() and sample_series.astype(str).str.contains('-').any():
                        df[col] = df[col].apply(
                            lambda x: f"'{x}" if pd.notnull(x) and str(x).startswith('+') and '-' in str(x) and not str(x).startswith("'") else x
                        )
        return df

    import pandas as pd
    import numpy as np

    if schema_columns is None:
        schema_columns = []

    # 1. تطبيق الصيغ الرياضية (formula) والشرطية (conditional) أولاً
    for rule in rules:
        rtype = rule.get("type")
        if rtype == "formula":
            target = rule.get("target")
            expr = rule.get("expression")
            if target and expr and target in df.columns:
                try:
                    df[target] = df.eval(expr)
                except Exception as e:
                    print(f"Formula evaluation failed for {target} = {expr}: {e}. Trying fallback execution.")
                    try:
                        local_dict = {col: df[col] for col in df.columns}
                        local_dict["np"] = np
                        df[target] = eval(expr, {"__builtins__": None}, local_dict)
                    except Exception as fallback_err:
                        print(f"Fallback formula evaluation failed: {fallback_err}")
        
        elif rtype == "conditional":
            if_col = rule.get("if_col")
            if_op = rule.get("if_op")
            if_val = rule.get("if_val")
            then_col = rule.get("then_col")
            then_val = rule.get("then_val")

            if if_col and if_op and if_val is not None and then_col and then_val is not None:
                if if_col in df.columns and then_col in df.columns:
                    try:
                        cond_expr = f"`{if_col}` {if_op} "
                        is_numeric_val = False
                        try:
                            float(if_val)
                            is_numeric_val = True
                        except ValueError:
                            pass
                        
                        if is_numeric_val:
                            cond_expr += str(if_val)
                            if not pd.api.types.is_numeric_dtype(df[if_col]):
                                df[if_col] = pd.to_numeric(df[if_col], errors='coerce')
                        else:
                            cond_expr += f"'{if_val}'"

                        mask = df.eval(cond_expr)

                        then_is_expr = False
                        if any(char in str(then_val) for char in ['+', '-', '*', '/', '%']) or any(c in str(then_val) for c in df.columns):
                            then_is_expr = True

                        if then_is_expr:
                            try:
                                val_series = df.eval(str(then_val))
                                df.loc[mask, then_col] = val_series[mask]
                            except Exception:
                                df.loc[mask, then_col] = then_val
                        else:
                            then_col_info = next((c for c in schema_columns if c.get("name") == then_col), {}) if schema_columns else {}
                            is_float = then_col_info.get("is_float", False)
                            is_num = then_col_info.get("type") == "number" or then_col_info.get("type") in ["integer", "float"]
                            if is_num:
                                try:
                                    parsed_val = float(then_val)
                                    if not is_float:
                                        parsed_val = int(round(parsed_val))
                                    df.loc[mask, then_col] = parsed_val
                                except ValueError:
                                    df.loc[mask, then_col] = then_val
                            else:
                                df.loc[mask, then_col] = then_val
                    except Exception as cond_err:
                        print(f"Conditional rule application failed: {cond_err}")

    # 2. تطبيق الارتباطات الإحصائية (correlation)
    for rule in rules:
        rtype = rule.get("type")
        if rtype == "correlation":
            col1 = rule.get("col1")
            col2 = rule.get("col2")
            direction = rule.get("direction", "positive")

            if col1 and col2 and col1 in df.columns and col2 in df.columns:
                if col1 != col2 and len(df) > 1:
                    try:
                        if not pd.api.types.is_numeric_dtype(df[col1]):
                            df[col1] = pd.to_numeric(df[col1], errors='coerce').fillna(0)
                        if not pd.api.types.is_numeric_dtype(df[col2]):
                            df[col2] = pd.to_numeric(df[col2], errors='coerce').fillna(0)

                        expected_negative = (direction == "negative")

                        sorted_ind_indices = df[col1].argsort().values
                        sorted_dep_values = np.sort(df[col2].values)
                        if expected_negative:
                            sorted_dep_values = sorted_dep_values[::-1]

                        aligned_dep = np.empty_like(sorted_dep_values)
                        aligned_dep[sorted_ind_indices] = sorted_dep_values

                        std_val = np.std(sorted_dep_values)
                        if std_val == 0:
                            std_val = 1.0
                        noise = np.random.normal(0, std_val * 0.10, size=len(df))
                        correlated_values = aligned_dep + noise

                        col2_info = next((c for c in schema_columns if c.get("name") == col2), {}) if schema_columns else {}
                        c_min = col2_info.get("min")
                        c_max = col2_info.get("max")
                        
                        try:
                            c_min = float(c_min) if c_min is not None else float(df[col2].min())
                            c_max = float(c_max) if c_max is not None else float(df[col2].max())
                        except Exception:
                            c_min = float(df[col2].min())
                            c_max = float(df[col2].max())

                        final_values = np.clip(correlated_values, c_min, c_max)

                        is_float = col2_info.get("is_float", False)
                        if not is_float:
                            df[col2] = np.round(final_values).astype(int)
                        else:
                            df[col2] = final_values
                            
                        print(f"Applied procedural correlation between {col1} and {col2} ({direction})")
                    except Exception as corr_err:
                        print(f"Correlation application failed: {corr_err}")

    # Data Formatting (Excel Protection Fix):
    # Prepend a single quote (') to columns containing phone numbers (starting with '+' and containing hyphens '-')
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_series = df[col].dropna()
            if not sample_series.empty and sample_series.astype(str).str.startswith('+').any() and sample_series.astype(str).str.contains('-').any():
                df[col] = df[col].apply(
                    lambda x: f"'{x}" if pd.notnull(x) and str(x).startswith('+') and '-' in str(x) and not str(x).startswith("'") else x
                )

    return df


def generate_data_from_schema(schema_columns: list, rules: list, num_rows: int, locale: str = "en_US", user_prompt: str = "", user_id: int = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    يولّد DataFrame اصطناعي بناءً على قائمة الأعمدة والأنواع المحددة.
    يدعم العلاقات والارتباطات الإحصائية بين المتغيرات.
    """
    df = None

    # محاولة التوليد باستخدام الذكاء الاصطناعي لكتابة كود توليد مترابط
    if user_prompt:
        try:
            df = generate_data_via_code_agent(user_prompt, schema_columns, rules, num_rows, locale, model="llama-3.3-70b-versatile", user_id=user_id, api_key=api_key)
            if df is not None and not df.empty:
                # التأكد من وجود كافة الأعمدة المطلوبة
                for col in schema_columns:
                    col_name = col.get("name", "column")
                    if col_name not in df.columns:
                        df[col_name] = None
        except Exception as e:
            import traceback
            import logging
            logger = logging.getLogger("synthetic_data")
            logger.error(f"Correlation Code Failed: {e}")
            traceback.print_exc()
            err_msg = str(e)
            print(f"Code agent generation failed, using standard generator fallback: {err_msg}")
            
            # If we get here, both primary and backup models failed. Log warning notification.
            if "429" in err_msg or "rate limit" in err_msg.lower() or "limit exceeded" in err_msg.lower() or "too many requests" in err_msg.lower():
                import re
                # Try to extract the cooldown duration
                cooldown_match = re.search(r"Please try again in ([a-zA-Z0-9\.]+)", err_msg)
                cooldown_duration = cooldown_match.group(1) if cooldown_match else "فترة قصيرة (cooldown period)"
                
                title = "تجاوز حد استخدام الذكاء الاصطناعي (Groq 429)"
                message = (
                    f"فشل توليد البيانات باستخدام وكيل الكود بسبب حد الاستخدام (429 Too Many Requests).\n"
                    f"تمت محاولة التبديل التلقائي لنموذج الاحتياطي (meta-llama/llama-4-scout-17b-16e-instruct) ولكنه فشل أيضاً.\n"
                    f"تم تشغيل المولد التقليدي كبديل لتفادي تعطل عملك.\n"
                    f"يرجى محاولة التوليد الذكي مجدداً بعد {cooldown_duration}، أو قم بتهيئة مفتاح API جديد في صفحة الإعدادات."
                )
                
                if user_id:
                    try:
                        from backend.database import SessionLocal
                        from backend.models import Notification
                        db = SessionLocal()
                        notification = Notification(
                            user_id=user_id,
                            title=title,
                            message=message,
                            type="warning"
                        )
                        db.add(notification)
                        db.commit()
                        db.close()
                        print(f"Logged 429 rate limit notification for user_id={user_id}")
                    except Exception as db_err:
                        print(f"Failed to log notification in DB: {db_err}")

    # Fallback procedural generation if AI generation failed or wasn't requested
    if df is None or df.empty:
        import random

        # تهيئة الـ Faker
        try:
            from faker import Faker
            fake_inst = Faker(locale)
        except Exception:
            fake_inst = FallbackFaker(locale)

        columns_data = {}
        col_generators = []

        for col in schema_columns:
            col_type = col.get("type", "text")
            col_name = col.get("name", "column")
            cats = col.get("categories")

            # 1. Direct custom categories selection
            if isinstance(cats, list) and len(cats) > 0:
                col_generators.append((col_name, lambda i, c=cats: random.choice(c)))
            # 2. Standard column types
            elif col_type == "id":
                columns_data[col_name] = list(range(1, num_rows + 1))
            elif col_type == "category":
                fallback_cats = ["Category A", "Category B", "Category C"]
                col_generators.append((col_name, lambda i, c=fallback_cats: random.choice(c)))
            elif col_type == "name":
                col_generators.append((col_name, lambda i: fake_inst.name()))
            elif col_type == "email":
                col_generators.append((col_name, lambda i: fake_inst.email()))
            elif col_type == "phone":
                col_generators.append((col_name, lambda i: fake_inst.phone_number()))
            elif col_type == "address":
                col_generators.append((col_name, lambda i: fake_inst.address().replace("\n", ", ")))
            elif col_type == "number":
                c_min = int(col.get("min", 0))
                c_max = int(col.get("max", 100))
                is_float = col.get("is_float", False)
                if is_float:
                    col_generators.append((col_name, lambda i, mn=c_min, mx=c_max: round(random.uniform(mn, mx), 2)))
                else:
                    col_generators.append((col_name, lambda i, mn=c_min, mx=c_max: random.randint(mn, mx)))
            elif col_type == "date":
                if hasattr(fake_inst, "date_between"):
                    col_generators.append((col_name, lambda i: fake_inst.date_between(start_date='-5y', end_date='today').strftime('%Y-%m-%d')))
                else:
                    col_generators.append((col_name, lambda i: f"{random.randint(2018, 2026)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"))
            else:  # text or other fallbacks
                col_name_lower = col_name.lower()
                if 'city' in col_name_lower:
                    if hasattr(fake_inst, "city"):
                        col_generators.append((col_name, lambda i: fake_inst.city()))
                    else:
                        col_generators.append((col_name, lambda i: fake_inst.address().split(',')[0]))
                elif 'email' in col_name_lower:
                    col_generators.append((col_name, lambda i: fake_inst.email()))
                elif 'name' in col_name_lower:
                    if any(kw in col_name_lower for kw in ['company', 'store', 'business', 'corp', 'brand', 'firm', 'vendor', 'supplier']):
                        if hasattr(fake_inst, "company"):
                            col_generators.append((col_name, lambda i: fake_inst.company()))
                        else:
                            col_generators.append((col_name, lambda i: fake_inst.name()))
                    else:
                        col_generators.append((col_name, lambda i: fake_inst.name()))
                elif 'phone' in col_name_lower or 'mobile' in col_name_lower or 'tel' in col_name_lower:
                    col_generators.append((col_name, lambda i: fake_inst.phone_number()))
                elif 'address' in col_name_lower or 'street' in col_name_lower or 'location' in col_name_lower:
                    col_generators.append((col_name, lambda i: fake_inst.address().replace("\n", ", ")))
                elif 'country' in col_name_lower:
                    if hasattr(fake_inst, "country"):
                        col_generators.append((col_name, lambda i: fake_inst.country()))
                    else:
                        col_generators.append((col_name, lambda i: fake_inst.address().split(',')[-1].strip()))
                elif col_type == "text":
                    if hasattr(fake_inst, "sentence"):
                        col_generators.append((col_name, lambda i: fake_inst.sentence(nb_words=10)))
                    else:
                        desc = "بيان نصي عشوائي للمحاكاة" if locale.startswith("ar") else "Random text simulation entry"
                        col_generators.append((col_name, lambda i: f"{desc} {random.randint(10, 99)}"))
                elif "note" in col_name_lower or "description" in col_name_lower or any(kw in col_name_lower for kw in ['comment', 'article', 'sentence', 'paragraph', 'about', 'bio']):
                    if hasattr(fake_inst, "sentence"):
                        col_generators.append((col_name, lambda i: fake_inst.sentence()))
                    else:
                        desc = "ملاحظة عشوائية للمحاكاة" if locale.startswith("ar") else "Random note simulation entry"
                        col_generators.append((col_name, lambda i: f"{desc} {random.randint(10, 99)}"))
                else:
                    if hasattr(fake_inst, "word"):
                        col_generators.append((col_name, lambda i: fake_inst.word()))
                    else:
                        col_generators.append((col_name, lambda i: f"Item_{random.randint(10, 99)}"))

        # Execute all generators dynamically per row
        for col_name, gen_func in col_generators:
            columns_data[col_name] = [gen_func(i) for i in range(num_rows)]

        df = pd.DataFrame(columns_data)

    # Ensure all columns in schema exist
    for col in schema_columns:
        col_name = col.get("name", "column")
        if col_name not in df.columns:
            df[col_name] = None

    # Safety net: lightly correlate related numerical columns if they exist (unconditional verification)
    ind_keywords = ['experience', 'years_of_experience', 'age', 'year', 'car_year', 'service_years', 'years', 'tenure']
    dep_keywords = ['salary', 'income', 'price', 'sales', 'annual_sales', 'revenue', 'cost', 'compensation']

    ind_col = None
    dep_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ind_keywords):
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
            if pd.api.types.is_numeric_dtype(df[col]):
                ind_col = col
                break

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in dep_keywords):
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
            if pd.api.types.is_numeric_dtype(df[col]):
                dep_col = col
                break

    if ind_col and dep_col and ind_col != dep_col and len(df) > 1:
        # Calculate current Pearson correlation coefficient
        try:
            current_corr = df[ind_col].corr(df[dep_col])
        except Exception:
            current_corr = np.nan

        # Enforce safety net if correlation is low or NaN
        if np.isnan(current_corr) or abs(current_corr) < 0.5:
            import logging
            logger = logging.getLogger("synthetic_data")
            warning_msg = f"Low correlation detected ({current_corr:.4f}). Enforcing mathematical safety-net..."
            logger.warning(warning_msg)
            print(warning_msg)

            # Determine direction of correlation based on name heuristics
            expected_negative = False
            neg_heuristics = ['remaining', 'depreciation', 'life_expectancy', 'unused']
            if any(kw in ind_col.lower() or kw in dep_col.lower() for kw in neg_heuristics):
                expected_negative = True

            # Rank-based alignment
            sorted_ind_indices = df[ind_col].argsort().values
            sorted_dep_values = np.sort(df[dep_col].values)
            if expected_negative:
                sorted_dep_values = sorted_dep_values[::-1]

            aligned_dep = np.empty_like(sorted_dep_values)
            aligned_dep[sorted_ind_indices] = sorted_dep_values

            # Add controlled noise (10% standard deviation) to make distribution realistic but highly correlated (> 0.6)
            std_val = np.std(sorted_dep_values)
            if std_val == 0:
                std_val = 1.0
            noise = np.random.normal(0, std_val * 0.10, size=len(df))
            correlated_values = aligned_dep + noise

            # Obtain bounds from column specs
            col_info = next((c for c in schema_columns if c.get("name") == dep_col), {})
            try:
                c_min = float(col_info.get("min", df[dep_col].min()))
                c_max = float(col_info.get("max", df[dep_col].max()))
            except Exception:
                c_min = float(df[dep_col].min())
                c_max = float(df[dep_col].max())

            final_values = np.clip(correlated_values, c_min, c_max)

            # Explicit integer rounding and casting if the column is defined as an integer
            is_float = col_info.get("is_float", False)
            if not is_float:
                df[dep_col] = np.round(final_values).astype(int)
            else:
                df[dep_col] = final_values

    # Apply logical rules & user connections (procedural enforcement as safety fallback)
    df = apply_logical_rules(df, rules, schema_columns)

    # Final Type Casting and Rounding cleanup loop
    for col_info in schema_columns:
        col_name = col_info.get("name")
        if col_name in df.columns:
            col_type = col_info.get("type", "")
            is_float = col_info.get("is_float", False)
            
            # Detect if it represents an integer or float strictly from schema definitions
            is_int = (
                col_type == "integer" or
                col_type == "id" or
                (col_type == "number" and not is_float)
            )
            is_flt = (
                col_type == "float" or
                (col_type == "number" and is_float)
            )
            
            if is_int:
                try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).round().astype(int)
                except Exception:
                    try:
                        df[col_name] = df[col_name].fillna(0).astype(int)
                    except Exception:
                        pass
            elif is_flt:
                try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').round(2)
                except Exception:
                    try:
                        df[col_name] = df[col_name].round(2)
                    except Exception:
                        pass

    # Select and order final output columns
    col_names = [col.get("name", "column") for col in schema_columns]
    return df[col_names]
