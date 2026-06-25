import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import importlib
import io
import os
import json
import threading

# استدعاء محرك التوليد من ملفك
synth = importlib.import_module("synthetic ")

# ─────────────────────────────────────────────
# LLM Helper — Groq API (Llama 3 / Mixtral)
# ─────────────────────────────────────────────

# (تم إخفاء الموديلات وتكوين الـ API لتصبح متكاملة كطبقة أساسية في الخلفية)

def _build_prompt(profile: dict, df_shape: tuple) -> str:
    """يبني الـ prompt المرسل للـ LLM."""
    cols_summary = []
    for col, info in profile.items():
        if col.startswith("__"):
            continue
        col_type = info.get("type", "unknown")
        null_pct = info.get("null_pct", 0)
        if col_type == "numerical":
            cols_summary.append(
                f"- '{col}' [رقمي]: mean={info.get('mean', 0):.2f}, "
                f"std={info.get('std', 0):.2f}, min={info.get('min', 0):.2f}, "
                f"max={info.get('max', 0):.2f}, قيم مفقودة={null_pct:.1f}%"
            )
        elif col_type == "categorical":
            top_vals = list(info.get("value_counts", {}).keys())[:3]
            n_cats = len(info.get("value_counts", {}))
            cols_summary.append(
                f"- '{col}' [فئوي]: عدد الفئات={n_cats}, أشهرها={top_vals}, قيم مفقودة={null_pct:.1f}%"
            )
        elif col_type == "sensitive":
            cols_summary.append(f"- '{col}' [حساس — سيُولَّد بـ Faker تلقائياً]")

    data_summary = "\n".join(cols_summary)

    return f"""أنت خبير متخصص في علم البيانات وتوليد البيانات الاصطناعية (Synthetic Data).
لديك مجموعة بيانات من {df_shape[0]} صف و {df_shape[1]} عمود تريد توليد نسخة اصطناعية منها.

هيكل الأعمدة وإحصائياتها:
{data_summary}

المطلوب — أجب بشكل مختصر وعملي:

1. **تقييم سريع**: جملة واحدة عن جودة هذه البيانات ومدى ملاءمتها للتوليد.

2. **طريقة التوليد الموصى بها**: هل CTGAN أم الطريقة الإحصائية (Basic)؟ ولماذا؟
   - إذا CTGAN، ما عدد الـ Epochs المناسب (50-500)؟

3. **تحذيرات الأعمدة**:
   - أي أعمدة فئوية تعاني من انحياز (imbalance) وتحتاج Class Balancing؟
   - أي أعمدة تحتوي على قيم مفقودة تحتاج انتباهاً؟

4. **توصية الضوضاء**: هل تنصح بحقن ضوضاء (Noise Injection)؟ بأي نسبة؟

اكتب إجابتك باللغة العربية فقط، بنقاط واضحة ومختصرة."""


def get_llm_analysis(profile: dict, df_shape: tuple) -> str:
    """
    يرسل ملخص هيكل البيانات إلى Groq API ويعيد توصيات ذكية.
    يعتمد على المفتاح المدمج والموديل الأساسي (Llama 3.3).
    """
    try:
        from groq import Groq
        key_to_use = synth.GROQ_API_KEY if synth.GROQ_API_KEY and synth.GROQ_API_KEY.startswith("gsk") else os.getenv("GROQ_API_KEY", "")
        if not key_to_use.startswith("gsk"):
            return "❌ تعذر الاتصال بـ الذكاء الاصطناعي: مفتاح Groq غير متوفر في الكود الأساسي (synthetic.py)."

        client = Groq(api_key=key_to_use)
        prompt = _build_prompt(profile, df_shape)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except ImportError:
        return "❌ يرجى تثبيت مكتبة Groq أولاً:\n```\npip install groq\n```"
    except Exception as e:
        err = str(e)
        if "invalid_api_key" in err.lower() or "401" in err:
            return "❌ المفتاح المدمج في النظام غير صحيح."
        elif "rate_limit" in err.lower() or "429" in err:
            return "⚠️ النظام مشغول حالياً (Rate Limit). انتظري قليلاً ثم حاولي مرة أخرى."
        return f"❌ خطأ داخلي في الاتصال بـ الذكاء الاصطناعي: {err}"


# ─────────────────────────────────────────────
# إعداد الصفحة
# ─────────────────────────────────────────────
st.set_page_config(page_title="SOL — Data Generator", layout="wide", page_icon="🧬")

st.title("🧬 محرك توليد البيانات الاصطناعية — SOL Platform")
st.markdown("قم برفع بياناتك الأصلية لإنشاء نسخة اصطناعية تحافظ على الخصائص الإحصائية بدون أي بيانات حقيقية حساسة.")

# إضافة تبويبات (Tabs) حسب المتطلبات
tab1, tab_eda, tab2 = st.tabs(["🚀 Data Generator", "🔍 استكشاف البيانات (EDA)", "ℹ️ حول النظام"])

with tab_eda:
    st.markdown("### 📊 استكشاف البيانات (Advanced EDA)")
    if "original_df" in st.session_state:
        df_eda = st.session_state["original_df"]
        st.markdown("هذه اللوحة الذكية تساعدك على فهم وتحليل بياناتك الأصلية بعمق قبل توليد البيانات الاصطناعية منها.")
        
        # 1. Dataset KPIs (مؤشرات الأداء)
        st.markdown("#### 📌 نظرة عامة على البيانات")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("إجمالي الصفوف", f"{len(df_eda):,}")
        kpi2.metric("إجمالي الأعمدة", f"{len(df_eda.columns):,}")
        kpi3.metric("الصفوف المكررة", f"{df_eda.duplicated().sum():,}")
        memory_usage = df_eda.memory_usage(deep=True).sum() / (1024 * 1024)
        kpi4.metric("حجم الذاكرة", f"{memory_usage:.2f} MB")
        
        st.markdown("---")
        
        # 2. القيم المفقودة والارتباطات (Missing Values & Correlation)
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("#### 🧩 القيم المفقودة (Missing Values)")
            missing = df_eda.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                fig_missing = px.bar(
                    missing, x=missing.index, y=missing.values, 
                    labels={'x': 'العمود', 'y': 'عدد القيم المفقودة'}, 
                    color=missing.values, color_continuous_scale='Reds'
                )
                fig_missing.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ بياناتك نظيفة تماماً! لا توجد قيم مفقودة.")
                
        with col_eda2:
            st.markdown("#### 🗺️ خريطة الارتباط الحرارية (Correlation Heatmap)")
            num_cols_eda = df_eda.select_dtypes(include=np.number).columns
            if len(num_cols_eda) > 1:
                corr_matrix = df_eda[num_cols_eda].corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale='RdBu_r'
                )
                fig_corr.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("ℹ️ يلزم وجود عمودين رقميين على الأقل لحساب الارتباط.")

        st.markdown("---")
        
        # 3. توزيع البيانات والتحليل الفئوي (Distributions & Categorical)
        st.markdown("#### 📈 التوزيع والتحليل العميق")
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            if len(num_cols_eda) > 0:
                selected_num = st.selectbox("اختر عموداً رقمياً لتحليل توزيعه:", num_cols_eda, key="eda_num_sel")
                # Histogram + Boxplot using plotly express marginal
                fig_dist = px.histogram(
                    df_eda, x=selected_num, 
                    marginal="box", 
                    color_discrete_sequence=['#00d4ff'],
                    title=f"توزيع الأرقام والقيم الشاذة لـ {selected_num}"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("لا توجد بيانات رقمية لعرض التوزيع.")
                
        with col_dist2:
            cat_cols_eda = df_eda.select_dtypes(exclude=np.number).columns
            if len(cat_cols_eda) > 0:
                selected_cat = st.selectbox("اختر عموداً نصياً لتحليل الفئات:", cat_cols_eda, key="eda_cat_sel")
                # Donut Chart
                val_counts = df_eda[selected_cat].value_counts().reset_index()
                val_counts.columns = [selected_cat, 'Count']
                fig_pie = px.pie(
                    val_counts, values='Count', names=selected_cat, 
                    hole=0.4, title=f"التوزيع الفئوي لـ {selected_cat}"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("لا توجد بيانات نصية/فئوية للتحليل.")
                
    else:
        st.warning("⚠️ يرجى رفع ملف البيانات في تبويب 'Data Generator' أولاً.")

with tab1:
    st.markdown("### 📂 خطوة 1: ارفع ملف البيانات الأصلي")
    uploaded_file = st.file_uploader("اختر ملف (CSV أو Excel)", type=["csv", "xlsx"])

    # ─── تصفير الذاكرة عند رفع ملف جديد ───
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        if st.session_state.get("last_uploaded_file") != current_file_name:
            # ملف جديد → نمسح البيانات القديمة
            for key in ["synthetic_df", "report_df", "privacy_report_df", "privacy_data",
                        "original_df", "profile", "gen_method_label"]:
                st.session_state.pop(key, None)
            st.session_state["last_uploaded_file"] = current_file_name

    if uploaded_file:
        try:
            # ─── قراءة الملف وتحليله ───
            if "original_df" not in st.session_state or "profile" not in st.session_state:
                with st.spinner("⚡ جاري قراءة الملف..."):
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                st.session_state["original_df"] = df

                # ─── تحليل الملف كاملاً ───
                n_rows_total = len(df)
                analysis_msg = f"🔬 جاري تحليل البيانات ({n_rows_total:,} صف × {len(df.columns)} عمود)..."

                with st.spinner(analysis_msg):
                    profile = synth.profile_dataframe(df)
                st.session_state["profile"] = profile

            else:
                df = st.session_state["original_df"]
                profile = st.session_state["profile"]

            # استخراج اسم الملف الأصلي بدون الصيغة (مثال: ires.csv → ires)
            base_name = os.path.splitext(uploaded_file.name)[0]

            # ─── رسالة النجاح ───
            meta = profile.get("__meta__", {})
            st.success(f"✅ تم رفع الملف وتحليله بنجاح! ({meta.get('total_rows', len(df)):,} صف × {len(df.columns)} عمود)")

            cat_cols = [c for c, v in profile.items() if isinstance(v, dict) and v.get("type") == "categorical"]
            num_cols = [c for c, v in profile.items() if isinstance(v, dict) and v.get("type") == "numerical"]

            with st.expander("👀 معاينة البيانات الأصلية", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # ─── مساعد الذكاء الاصطناعي (LLM Advisor) ───
            st.markdown("---")
            st.markdown("### 🤖 خطوة 2 (اختياري): مساعد الذكاء الاصطناعي")
            with st.expander("🧠 تحليل البيانات واستخراج التوصيات بالذكاء الاصطناعي", expanded=False):
                st.info("يقوم الذكاء الاصطناعي المدمج بفحص دقيق لبياناتك واقتراح أفضل مسار لتوليد البيانات الاصطناعية بأعلى جودة.")
                
                analyze_btn = st.button("🔍 حلّل بياناتي الآن", use_container_width=True, type="primary")

                if analyze_btn:
                    with st.spinner("🤖 جاري التحليل والفهم العميق... ⚡"):
                        llm_result = get_llm_analysis(profile, df.shape)
                    st.session_state["llm_advice"] = llm_result

                if "llm_advice" in st.session_state:
                    st.markdown("---")
                    st.markdown("#### 💡 توصيات الذكاء الاصطناعي:")
                    advice = st.session_state["llm_advice"]
                    if advice.startswith("❌") or advice.startswith("⚠️"):
                        st.error(advice)
                    else:
                        st.success(advice)

            # ─── التحكم الدقيق وقواعد التحقق (Data Validation) ───
            st.markdown("---")
            st.markdown("### 🛠️ خطوة 3: التحكم الدقيق وقواعد التحقق")
            with st.expander("⚙️ تعديل أنواع الأعمدة ووضع قواعد (استبعاد، تغيير نوع، حدود رقمية)", expanded=False):
                editor_data = []
                for c, v in profile.items():
                    if c.startswith("__"): continue
                    editor_data.append({
                        "العمود": c,
                        "استبعاد": False,
                        "النوع المكتشف": v["type"],
                        "الحد الأدنى": float(v.get("min", 0)) if v["type"] == "numerical" else None,
                        "الحد الأقصى": float(v.get("max", 0)) if v["type"] == "numerical" else None,
                    })
                editor_df = pd.DataFrame(editor_data)
                
                edited_df = st.data_editor(
                    editor_df,
                    column_config={
                        "استبعاد": st.column_config.CheckboxColumn(default=False),
                        "النوع المكتشف": st.column_config.SelectboxColumn(
                            options=["numerical", "categorical", "id_sequence", "pattern_sequence", "sensitive_name", "sensitive_email", "sensitive_phone", "sensitive_address", "sensitive_id"]
                        ),
                        "الحد الأدنى": st.column_config.NumberColumn(),
                        "الحد الأقصى": st.column_config.NumberColumn(),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # تطبيق التعديلات على الـ profile محلياً
                active_cols = []
                for _, row in edited_df.iterrows():
                    c = row["العمود"]
                    if not row["استبعاد"]:
                        active_cols.append(c)
                        profile[c]["type"] = row["النوع المكتشف"]
                        if profile[c]["type"] == "numerical":
                            if pd.notnull(row["الحد الأدنى"]): profile[c]["min"] = row["الحد الأدنى"]
                            if pd.notnull(row["الحد الأقصى"]): profile[c]["max"] = row["الحد الأقصى"]

                df_to_generate = df[active_cols].copy()

            st.markdown("---")
            st.markdown("### ⚙️ خطوة 4: إعدادات التوليد الأساسية")

            col_set1, col_set2, col_set3 = st.columns(3)

            with col_set1:
                gen_mode = st.radio("وضع التوليد:", ["🔄 استبدال (Replacement)", "➕ توسيع (Augmentation)"], horizontal=True)
                
                if "توسيع" in gen_mode:
                    st.caption("📈 **النتيجة:** دمج البيانات الوهمية الجديدة مع بياناتك الأصلية (مثالي لتضخيم الحجم لتدريب الذكاء الاصطناعي).")
                    n_rows = st.number_input("عدد الصفوف الإضافية:", min_value=1, value=len(df), step=100)
                else:
                    st.caption("✨ **النتيجة:** ملف جديد وهمي 100% لا يحتوي على أي بيانات أصلية (مثالي للمشاركة الآمنة).")
                    n_rows = st.number_input("عدد الصفوف المطلوب توليدها:", min_value=1, value=len(df), step=100)

            with col_set2:
                method = st.selectbox(
                    "طريقة التوليد:",
                    [
                        "🧠 ذكاء اصطناعي سريع (TVAE — ثوانٍ)",
                        "⚡ سريع (GaussianCopula — ثوانٍ)",
                        "🧠 جودة عالية (CTGAN — دقائق)",
                        "📊 إحصائية (Basic)",
                        "🏆 مقارنة تلقائية (Auto-Benchmark)",
                    ],
                    help="TVAE: نموذج ذكاء اصطناعي سريع وموثوق | GaussianCopula: نموذج إحصائي سريع | CTGAN: دقة عالية ولكن بطيء جداً."
                )

            with col_set3:
                if "CTGAN" in method or "TVAE" in method or "Auto-Benchmark" in method:
                    epochs = st.slider(
                        "عدد دورات التدريب (Epochs):",
                        min_value=10, max_value=200, value=30, step=10,
                        help="30 epoch كافٍ لأغلب الحالات. زيادته تزيد الدقة لكن تبطئ التدريب."
                    )
                    train_sample_size = st.number_input(
                        "حجم عينة التدريب:",
                        min_value=100, max_value=len(df),
                        value=min(2000, len(df)),
                        step=500,
                        help="تقليل العينة يسرع تدريب الذكاء الاصطناعي بشكل كبير."
                    )
                else:
                    st.info("الطريقة المختارة لا تحتاج إعدادات إضافية.")
                    epochs = 30
                    train_sample_size = 2000

            st.markdown("---")

            with st.expander("🧪 إعدادات متقدمة: التوجيه ومختبر الضوضاء (اختياري)", expanded=False):
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    st.markdown("**1. التوليد الموجه (Class Balancing):**")
                    balance_col = st.selectbox("اختر عموداً فئوياً لمساواة التوزيع (اختياري):", ["لا شيء"] + cat_cols)
                    if balance_col != "لا شيء":
                        st.info(f"النسب الأصلية لـ {balance_col}:")
                        st.dataframe(df[balance_col].value_counts(normalize=True).mul(100).round(1).astype(str) + '%', height=100)
                        equalize = st.checkbox(f"مساواة جميع الفئات لعمود '{balance_col}' بالتساوي (Equal Split)")
                    else:
                        equalize = False

                with adv_col2:
                    st.markdown("**2. مختبر الضوضاء (Noise Injection):**")
                    st.markdown("مفيد لإنشاء بيانات غير مثالية (Dirty Data) لاختبار نماذج الذكاء الاصطناعي ومرونة الأنظمة.")
                    inject_nulls = st.slider("نسبة حقن القيم المفقودة (Nulls):", 0, 50, 0, format="%d%%")
                    inject_outliers = st.slider("نسبة حقن القيم المتطرفة (Outliers):", 0, 30, 0, format="%d%%")

                st.markdown("**3. شروط مخصصة (Custom Constraints):**")
                custom_query = st.text_input(
                    "اكتب شرطاً لتصفية البيانات (Pandas Query):",
                    placeholder="مثال: Age >= 18 and Salary > 2000",
                    help="سيتم استبعاد أي صف لا يطابق هذا الشرط. يجب كتابة أسماء الأعمدة بالإنجليزية لتجنب الأخطاء."
                )

            st.markdown("---")

            if st.button("✨ Generate Synthetic Version", use_container_width=True, type="primary"):

                # تطبيق التوجيه (Balancing) إذا طُلب في الطريقة الإحصائية
                if equalize and balance_col != "لا شيء":
                    if "Basic" in method or "Auto-Benchmark" in method:
                        cats = list(profile[balance_col]["value_counts"].keys())
                        eq_weight = 1.0 / len(cats)
                        profile[balance_col]["value_counts"] = {c: eq_weight for c in cats}
                        st.toast(f"تم مساواة توزيع {balance_col} بنجاح للأسلوب الإحصائي!", icon="⚖️")
                    else:
                        st.warning("⚠️ التوجيه المباشر غير مدعوم في نماذج الذكاء الاصطناعي حالياً.")

                def do_generate_basic():
                    return synth.generate_synthetic(df_to_generate, profile, n_rows)

                def do_generate_fast():
                    progress_bar = st.progress(0, text="جاري البدء بالنموذج السريع...")
                    def update_progress(pct, msg):
                        progress_bar.progress(pct, text=msg)
                    res = synth.generate_synthetic_fast(df_to_generate, profile, n_rows, progress_callback=update_progress)
                    progress_bar.empty()
                    return res

                def do_generate_tvae():
                    progress_bar = st.progress(0, text="جاري البدء بـ TVAE...")
                    def update_progress(pct, msg):
                        progress_bar.progress(pct, text=msg)
                    res = synth.generate_synthetic_tvae(df_to_generate, profile, n_rows, epochs=epochs, train_sample_size=train_sample_size, progress_callback=update_progress)
                    progress_bar.empty()
                    return res

                def do_generate_ai():
                    progress_bar = st.progress(0, text="جاري البدء بالذكاء الاصطناعي...")
                    def update_progress(pct, msg):
                        progress_bar.progress(pct, text=msg)
                    res = synth.generate_synthetic_ai(df_to_generate, profile, n_rows, epochs=epochs, train_sample_size=train_sample_size, progress_callback=update_progress)
                    progress_bar.empty()
                    return res

                if "Auto-Benchmark" in method:
                    st.info("🏆 جاري تنفيذ المقارنة التلقائية (Auto-Benchmark)...")
                    with st.spinner("جاري التوليد بالطريقة الإحصائية..."):
                        df_basic = do_generate_basic()
                    
                    df_ai = do_generate_tvae()
                    
                    # Evaluate both (باستخدام generate_report)
                    report_basic = synth.generate_report(df_to_generate, df_basic, profile)
                    score_basic = report_basic.iloc[-1]["Fidelity Score (%)"]
                    
                    report_ai = synth.generate_report(df_to_generate, df_ai, profile)
                    score_ai = report_ai.iloc[-1]["Fidelity Score (%)"]
                    
                    if score_ai >= score_basic:
                        st.success(f"🤖 فاز الذكاء الاصطناعي السريع! (TVAE: {score_ai}% vs Basic: {score_basic}%)")
                        synthetic_df = df_ai
                        gen_method_label = "🧠 TVAE (فائز بالمقارنة)"
                    else:
                        st.success(f"📊 فازت الطريقة الإحصائية! (Basic: {score_basic}% vs TVAE: {score_ai}%)")
                        synthetic_df = df_basic
                        gen_method_label = "📊 Basic (فائز بالمقارنة)"
                        
                elif "CTGAN" in method:
                    synthetic_df = do_generate_ai()
                    gen_method_label = "🧠 CTGAN (ذكاء اصطناعي)"
                elif "TVAE" in method:
                    synthetic_df = do_generate_tvae()
                    gen_method_label = "🧠 TVAE (ذكاء اصطناعي سريع)"
                elif "GaussianCopula" in method:
                    synthetic_df = do_generate_fast()
                    gen_method_label = "⚡ GaussianCopula (نموذج إحصائي سريع)"
                else:
                    with st.spinner("جاري التحليل الإحصائي وتوليد البيانات بالتوازي (Threads)... ⚡"):
                        synthetic_df = do_generate_basic()
                    gen_method_label = "📊 إحصائية (Basic)"
                    
                # ─── Data Augmentation Mode ───
                if "توسيع" in gen_mode:
                    # دمج البيانات الأصلية مع المولدة (للأعمدة النشطة فقط)
                    synthetic_df = pd.concat([df_to_generate, synthetic_df], ignore_index=True)
                    gen_method_label += " + ➕ وضع التوسيع"

                # 2. حقن الضوضاء (إن وُجدت)
                if inject_nulls > 0 or inject_outliers > 0:
                    with st.spinner("جاري حقن الضوضاء والقيم المتطرفة... 🦠"):
                        synthetic_df = synth.inject_noise(synthetic_df, profile, inject_nulls, inject_outliers)

                # 2.5 تطبيق الشروط المخصصة
                if custom_query:
                    try:
                        synthetic_df = synthetic_df.query(custom_query)
                        if len(synthetic_df) == 0:
                            st.error("⚠️ الشرط أدى إلى استبعاد جميع البيانات! سيتم عرض البيانات بدون الشرط.")
                            if "Basic" in method:
                                synthetic_df = synth.generate_synthetic(df, profile, n_rows)
                            elif "TVAE" in method:
                                synthetic_df = synth.generate_synthetic_tvae(df, profile, n_rows, epochs=epochs, train_sample_size=train_sample_size)
                            elif "GaussianCopula" in method:
                                synthetic_df = synth.generate_synthetic_fast(df, profile, n_rows)
                            else:
                                synthetic_df = synth.generate_synthetic_ai(df, profile, n_rows, epochs=epochs, train_sample_size=train_sample_size)
                        else:
                            st.toast(f"✅ تم تطبيق الشرط بنجاح! تبقى {len(synthetic_df)} صف.", icon="✅")
                    except Exception as e:
                        st.error(f"❌ خطأ في صياغة الشرط: {e}")

                # 3. تقرير الجودة — يقارن فقط الأعمدة النشطة (المولدة فعلياً)
                # نستخدم df_to_generate بدلاً من df الكامل لتجنب KeyError عند استبعاد أعمدة
                orig_for_report = df_to_generate.copy()
                # نضيف فقط الأعمدة الموجودة في synthetic_df
                available_cols = [c for c in orig_for_report.columns if c in synthetic_df.columns]
                orig_for_report = orig_for_report[available_cols]
                synthetic_for_report = synthetic_df[available_cols]
                report_df = synth.generate_report(orig_for_report, synthetic_for_report, profile)

                # 4. تحليل الخصوصية
                privacy_report_df, privacy_data = synth.generate_privacy_report(orig_for_report, synthetic_for_report, profile)

                # ─── حفظ كل النتائج في session_state ───
                st.session_state["synthetic_df"] = synthetic_df
                st.session_state["report_df"] = report_df
                st.session_state["privacy_report_df"] = privacy_report_df
                st.session_state["privacy_data"] = privacy_data
                st.session_state["gen_method_label"] = gen_method_label
                st.session_state["base_name"] = base_name

            # ─── عرض النتائج من session_state (تبقى حتى بعد التحميل) ───
            if "synthetic_df" in st.session_state:
                synthetic_df = st.session_state["synthetic_df"]
                report_df = st.session_state["report_df"]
                privacy_report_df = st.session_state["privacy_report_df"]
                privacy_data = st.session_state["privacy_data"]
                gen_method_label = st.session_state.get("gen_method_label", "")
                saved_base_name = st.session_state.get("base_name", base_name)

                st.success("🎉 تم الانتهاء من التوليد بنجاح!")

                # --- البطاقات التعريفية (Metrics) ---
                avg_score_val = report_df[report_df['العمود'] == '── المتوسط الكلي ──']['Fidelity Score (%)'].values
                avg_score = avg_score_val[0] if len(avg_score_val) > 0 else 0

                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                col_m1.metric("📄 الصفوف الأصلية", f"{len(df):,}")
                col_m2.metric("🧬 الصفوف المولدة", f"{len(synthetic_df):,}")
                col_m3.metric("🎯 متوسط التشابه", f"{avg_score}%",
                              "جودة عالية ✅" if avg_score > 80 else "جودة متوسطة ⚠️")
                col_m4.metric("🔒 نسبة الخصوصية", f"{privacy_data['privacy_score']}%",
                              privacy_data['risk_level'])
                col_m5.metric("⚡ طريقة التوليد", 
                              "CTGAN" if "CTGAN" in gen_method_label else 
                              "TVAE" if "TVAE" in gen_method_label else 
                              "GaussianCopula" if "GaussianCopula" in gen_method_label else 
                              "Basic")

                st.markdown("---")

                # --- تبويبات النتائج ---
                res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
                    "📊 البيانات المولدة والتحميل",
                    "📈 تقارير الجودة والإحصائيات",
                    "👁️ المقارنة البصرية والارتباطات",
                    "🔒 تحليل الخصوصية (Privacy)"
                ])

                # ==========================================
                # تبويب 1: البيانات المولدة والتحميل
                # ==========================================
                with res_tab1:
                    st.markdown("### البيانات المولدة (Synthetic Data)")
                    st.dataframe(synthetic_df.head(100), use_container_width=True)

                    st.markdown("### 📥 خيارات التحميل والتصدير")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv = synthetic_df.to_csv(index=False).encode('utf-8-sig')
                        csv_filename = f"SOL-{saved_base_name}_synthetic.csv"
                        st.download_button(
                            label=f"تحميل البيانات المولدة ({csv_filename}) 📄",
                            data=csv,
                            file_name=csv_filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                    with col_dl2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            synthetic_df.to_excel(writer, sheet_name='Synthetic Data', index=False)
                            report_df.to_excel(writer, sheet_name='Fidelity Report', index=False)
                            privacy_report_df.to_excel(writer, sheet_name='Privacy Report', index=False)
                            active_num_cols_dl = [c for c in num_cols if c in synthetic_df.columns]
                            if active_num_cols_dl:
                                df[active_num_cols_dl].describe().T.to_excel(writer, sheet_name='Original Stats')
                                synthetic_df[active_num_cols_dl].describe().T.to_excel(writer, sheet_name='Synthetic Stats')
                            if len(active_num_cols_dl) > 1:
                                df[active_num_cols_dl].corr().to_excel(writer, sheet_name='Original Correlations')
                                synthetic_df[active_num_cols_dl].corr().to_excel(writer, sheet_name='Synthetic Correlations')

                        excel_filename = f"SOL-{saved_base_name}_comprehensive_report.xlsx"
                        st.download_button(
                            label=f"تحميل تقرير شامل ({excel_filename}) 🌟",
                            data=buffer.getvalue(),
                            file_name=excel_filename,
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )

                    st.markdown("---")
                    st.markdown("### 📚 قاموس البيانات (Data Dictionary)")
                    col_dict1, _ = st.columns([1, 1])
                    with col_dict1:
                        md_dict = synth.generate_data_dictionary(st.session_state["profile"], synthetic_df)
                        st.download_button(
                            label="تحميل قاموس البيانات (Markdown) 📖",
                            data=md_dict.encode("utf-8"),
                            file_name=f"SOL-{saved_base_name}_DataDictionary.md",
                            mime="text/markdown",
                            use_container_width=True
                        )

                    st.markdown("---")
                    st.markdown("### 📋 معاينة ومشاركة البيانات (JSON)")
                    st.markdown("يمكنك تحميل البيانات بصيغة JSON لاستخدامها مباشرة في أي تطبيق أو مشاركتها مع المطورين.")

                    col_json1, col_json2 = st.columns([1, 1])
                    with col_json1:
                        json_bytes = synthetic_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
                        json_filename = f"SOL-{saved_base_name}_data.json"
                        st.download_button(
                            label=f"⬇️ تحميل البيانات ({json_filename})",
                            data=json_bytes,
                            file_name=json_filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    with col_json2:
                        json_size_kb = len(json_bytes) / 1024
                        st.metric("📦 حجم الملف", f"{json_size_kb:.1f} KB")

                    with st.expander("👁️ معاينة البيانات بصيغة JSON (أول 5 سجلات)", expanded=False):
                        st.json(synthetic_df.head(5).to_dict(orient="records"))

                # ==========================================
                # تبويب 2: تقارير الجودة والإحصائيات
                # ==========================================
                with res_tab2:
                    st.markdown("### تقرير نسبة التشابه لكل عمود (Fidelity Score)")
                    st.dataframe(report_df, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 📋 ملخص إحصائي مقارن (للمتغيرات الرقمية)")
                    # تصفية الأعمدة لتشمل فقط ما هو موجود فعلاً في synthetic_df
                    active_num_cols = [c for c in num_cols if c in synthetic_df.columns]
                    active_cat_cols = [c for c in cat_cols if c in synthetic_df.columns]
                    if active_num_cols:
                        for col_name in active_num_cols:
                            with st.expander(f"📊 {col_name}", expanded=False):
                                comp_col1, comp_col2 = st.columns(2)
                                with comp_col1:
                                    st.markdown("**البيانات الأصلية:**")
                                    orig_stats = df[col_name].describe()
                                    st.dataframe(orig_stats.to_frame().T, use_container_width=True)
                                with comp_col2:
                                    st.markdown("**البيانات المولدة:**")
                                    synth_stats = synthetic_df[col_name].describe()
                                    st.dataframe(synth_stats.to_frame().T, use_container_width=True)
                    else:
                        st.info("لا توجد أعمدة رقمية نشطة لعرض ملخصها الإحصائي.")

                    st.markdown("---")
                    st.markdown("### 📋 ملخص المتغيرات الفئوية (Categorical)")
                    if active_cat_cols:
                        for col_name in active_cat_cols:
                            with st.expander(f"📂 {col_name}", expanded=False):
                                comp_col1, comp_col2 = st.columns(2)
                                with comp_col1:
                                    st.markdown("**التوزيع الأصلي:**")
                                    st.dataframe(
                                        df[col_name].value_counts(normalize=True).reset_index().rename(
                                            columns={col_name: "القيمة", "proportion": "النسبة"}
                                        ), use_container_width=True
                                    )
                                with comp_col2:
                                    st.markdown("**التوزيع المولد:**")
                                    st.dataframe(
                                        synthetic_df[col_name].value_counts(normalize=True).reset_index().rename(
                                            columns={col_name: "القيمة", "proportion": "النسبة"}
                                        ), use_container_width=True
                                    )
                    else:
                        st.info("لا توجد أعمدة فئوية نشطة لعرض ملخصها.")

                # ==========================================
                # تبويب 3: المقارنة البصرية والارتباطات
                # ==========================================
                with res_tab3:
                    import plotly.graph_objects as go
                    import plotly.express as px
                    from plotly.subplots import make_subplots

                    st.markdown("### 📊 لوحة المقارنة البصرية التفاعلية")
                    st.markdown("جميع الرسومات تفاعلية — يمكنك التمرير فوقها ولمس النقاط لمعرفة القيم بالتفصيل.")

                    # ── قسم 1: Radar Chart لدرجات الجودة ──────────────────────
                    st.markdown("---")
                    st.markdown("#### 🎯 مؤشر جودة كل عمود (Fidelity Radar)")
                    st.caption("كلما اقتربت القيمة من الحافة الخارجية (100%)، كان العمود أكثر تشابهاً مع الأصل.")

                    radar_data = report_df[report_df["العمود"] != "── المتوسط الكلي ──"]
                    if not radar_data.empty:
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=radar_data["Fidelity Score (%)"].tolist() + [radar_data["Fidelity Score (%)"].iloc[0]],
                            theta=radar_data["العمود"].tolist() + [radar_data["العمود"].iloc[0]],
                            fill='toself',
                            name='Fidelity Score',
                            line_color='#00d4ff',
                            fillcolor='rgba(0, 212, 255, 0.2)',
                            marker=dict(size=8, color='#00d4ff')
                        ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100],
                                                tickfont=dict(size=10),
                                                gridcolor='rgba(255,255,255,0.1)'),
                                angularaxis=dict(tickfont=dict(size=11))
                            ),
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=420,
                            margin=dict(t=30, b=30)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                    # ── قسم 2: Histogram + KDE للأعمدة الرقمية ───────────────
                    if num_cols:
                        st.markdown("---")
                        st.markdown("#### 📈 مقارنة التوزيع (Histogram مزدوج)")
                        st.caption("الأزرق = البيانات الأصلية | البرتقالي = البيانات المولدة | كلما تداخلا أكثر، زادت الجودة.")

                        sel_num_col = st.selectbox("اختر عمود رقمي:", num_cols, key="hist_sel")

                        orig_vals = df[sel_num_col].dropna()
                        synth_vals = synthetic_df[sel_num_col].dropna()

                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=orig_vals, name="البيانات الأصلية",
                            opacity=0.65, nbinsx=30,
                            marker_color='#4C9BE8',
                            histnorm='probability density'
                        ))
                        fig_hist.add_trace(go.Histogram(
                            x=synth_vals, name="البيانات المولدة",
                            opacity=0.65, nbinsx=30,
                            marker_color='#F4845F',
                            histnorm='probability density'
                        ))
                        fig_hist.update_layout(
                            barmode='overlay',
                            xaxis_title=sel_num_col,
                            yaxis_title="الكثافة",
                            legend=dict(orientation="h", y=1.1),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(20,20,35,0.5)',
                            height=380,
                            margin=dict(t=20, b=20),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # ── Box Plot مقارنة ──────────────────────────────────
                        st.markdown("#### 📦 Box Plot مقارن (توزيع + وسيط + شاذات)")
                        st.caption("يُظهر الوسيط والربعيلات وأي قيم شاذة في كل مجموعة بيانات.")

                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=orig_vals, name="الأصلية",
                            marker_color='#4C9BE8',
                            boxmean='sd',
                            jitter=0.3, pointpos=-1.8,
                            boxpoints='outliers'
                        ))
                        fig_box.add_trace(go.Box(
                            y=synth_vals, name="المولدة",
                            marker_color='#F4845F',
                            boxmean='sd',
                            jitter=0.3, pointpos=-1.8,
                            boxpoints='outliers'
                        ))
                        fig_box.update_layout(
                            yaxis_title=sel_num_col,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(20,20,35,0.5)',
                            height=380,
                            margin=dict(t=20, b=20),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                        )
                        st.plotly_chart(fig_box, use_container_width=True)

                    # ── قسم 3: Bar Chart للأعمدة الفئوية ─────────────────────
                    if cat_cols:
                        st.markdown("---")
                        st.markdown("#### 🏷️ مقارنة التوزيع الفئوي (Grouped Bar)")
                        st.caption("يُقارن نسب الفئات في البيانات الأصلية مقابل المولدة لكل قيمة.")

                        sel_cat_col = st.selectbox("اختر عمود فئوي:", cat_cols, key="cat_sel")

                        orig_counts = df[sel_cat_col].value_counts(normalize=True).reset_index()
                        orig_counts.columns = ["الفئة", "النسبة"]
                        orig_counts["المصدر"] = "الأصلية"

                        synth_counts = synthetic_df[sel_cat_col].value_counts(normalize=True).reset_index()
                        synth_counts.columns = ["الفئة", "النسبة"]
                        synth_counts["المصدر"] = "المولدة"

                        cat_combined = pd.concat([orig_counts, synth_counts])

                        fig_bar = px.bar(
                            cat_combined, x="الفئة", y="النسبة", color="المصدر",
                            barmode="group",
                            color_discrete_map={"الأصلية": "#4C9BE8", "المولدة": "#F4845F"},
                            text_auto=".1%"
                        )
                        fig_bar.update_layout(
                            xaxis_title=sel_cat_col,
                            yaxis_title="النسبة المئوية",
                            legend=dict(orientation="h", y=1.1),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(20,20,35,0.5)',
                            height=380,
                            margin=dict(t=20, b=20),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # ── قسم 4: Correlation Heatmaps بـ Plotly ─────────────────
                    if len(num_cols) > 1:
                        st.markdown("---")
                        st.markdown("#### 🔗 مصفوفة الارتباط (Correlation Heatmap تفاعلي)")
                        st.caption("اللون الأحمر = ارتباط موجب قوي | الأزرق = ارتباط سالب | كلما تشابهت الخريطتان، كانت جودة التوليد أعلى.")

                        orig_corr = df[num_cols].corr().round(3)
                        synth_corr = synthetic_df[num_cols].corr().round(3)
                        diff_corr = (orig_corr - synth_corr).abs().round(3)

                        fig_corr = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=["🔵 الأصلية", "🟠 المولدة", "🔴 فرق الارتباطات"],
                            horizontal_spacing=0.08
                        )

                        def make_heatmap(corr_matrix, colorscale, col_idx):
                            return go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns.tolist(),
                                y=corr_matrix.index.tolist(),
                                colorscale=colorscale,
                                zmin=-1, zmax=1,
                                text=corr_matrix.values.round(2),
                                texttemplate="%{text}",
                                textfont=dict(size=9),
                                showscale=(col_idx == 3)
                            )

                        fig_corr.add_trace(make_heatmap(orig_corr, 'RdBu_r', 1), row=1, col=1)
                        fig_corr.add_trace(make_heatmap(synth_corr, 'RdBu_r', 2), row=1, col=2)
                        fig_corr.add_trace(go.Heatmap(
                            z=diff_corr.values,
                            x=diff_corr.columns.tolist(),
                            y=diff_corr.index.tolist(),
                            colorscale='Reds', zmin=0, zmax=0.5,
                            text=diff_corr.values.round(3),
                            texttemplate="%{text}",
                            textfont=dict(size=9),
                            showscale=True
                        ), row=1, col=3)

                        fig_corr.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=420,
                            margin=dict(t=50, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                        # استخدام nanmean لتجنب قيم NaN الناتجة عن الأعمدة ذات القيم الثابتة (Variance = 0)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            avg_corr_diff = np.nanmean(diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)])

                        if pd.isnull(avg_corr_diff):
                            st.info("ℹ️ تعذر حساب متوسط فرق الارتباطات لأن بعض الأعمدة الرقمية في العينة تحتوي على قيم ثابتة بدون تغيير (Variance = 0).")
                        elif avg_corr_diff < 0.1:
                            st.success(f"✅ متوسط فرق الارتباطات: {avg_corr_diff:.4f} — ممتاز! العلاقات محفوظة بدقة عالية.")
                        elif avg_corr_diff < 0.25:
                            st.warning(f"⚠️ متوسط فرق الارتباطات: {avg_corr_diff:.4f} — مقبول. جرب زيادة عدد الـ Epochs.")
                        else:
                            st.error(f"❌ متوسط فرق الارتباطات: {avg_corr_diff:.4f} — ضعيف. يُنصح باستخدام CTGAN.")



                # ==========================================
                # تبويب 4: تحليل الخصوصية (Privacy)
                # ==========================================
                with res_tab4:
                    st.markdown("### 🔒 تحليل الخصوصية — Distance to Closest Record (DCR)")
                    st.markdown("""
                    يقيس هذا التحليل **المسافة بين كل صف مولد وأقرب صف حقيقي** في البيانات الأصلية.
                    الهدف هو التأكد من أن البيانات المولدة **جديدة تماماً** ولم تقم بنسخ أو تسريب أي بيانات حقيقية.
                    هذا المقياس ضروري للتوافق مع **قوانين حماية البيانات (GDPR)**.
                    """)

                    st.markdown("---")

                    priv_col1, priv_col2, priv_col3 = st.columns(3)
                    priv_col1.metric("🔒 نسبة الخصوصية", f"{privacy_data['privacy_score']}%")
                    priv_col2.metric("⚠️ صفوف معرضة للخطر", f"{privacy_data['at_risk_count']} / {privacy_data['total_checked']}")
                    priv_col3.metric("🛡️ مستوى الأمان", privacy_data['risk_level'])

                    st.markdown("---")

                    st.markdown("### 📋 تقرير الخصوصية التفصيلي")
                    st.dataframe(privacy_report_df, use_container_width=True)

                    st.markdown("---")

                    if len(privacy_data['dcr_values']) > 0:
                        st.markdown("### 📊 توزيع المسافات (DCR Distribution)")
                        st.markdown("كلما كان التوزيع مائلاً لليمين (مسافات أعلى)، كانت الخصوصية أفضل.")

                        dcr_hist, dcr_bins = np.histogram(privacy_data['dcr_values'], bins=30)
                        dcr_chart_df = pd.DataFrame({
                            'عدد الصفوف': dcr_hist
                        }, index=np.round(dcr_bins[:-1], 3))
                        st.bar_chart(dcr_chart_df)

                        st.markdown(f"🔴 **حد الخطر (Risk Threshold):** `{privacy_data['risk_threshold']}` — أي صف مسافته أقل من هذا الحد يُعتبر مشبوهاً.")

                    st.markdown("---")

                    st.markdown("### 💡 ماذا تعني هذه النتيجة؟")
                    if privacy_data['privacy_score'] >= 95:
                        st.success("""
                        ✅ **ممتاز!** البيانات المولدة آمنة تماماً. لا يوجد أي صف مولد يشبه صفاً حقيقياً بشكل خطير.
                        البيانات صالحة للاستخدام في بيئات الإنتاج وتتوافق مع معايير GDPR.
                        """)
                    elif privacy_data['privacy_score'] >= 80:
                        st.success("""
                        ✅ **آمن.** نسبة ضئيلة جداً من الصفوف قد تكون قريبة من البيانات الأصلية،
                        لكن بشكل عام البيانات آمنة للاستخدام.
                        """)
                    elif privacy_data['privacy_score'] >= 60:
                        st.warning("""
                        ⚠️ **مقبول.** بعض الصفوف المولدة قريبة من البيانات الأصلية.
                        يُنصح بزيادة عدد الـ Epochs أو استخدام طريقة CTGAN لتحسين التنوع.
                        """)
                    else:
                        st.error("""
                        🔴 **تحذير!** نسبة كبيرة من البيانات المولدة تشبه البيانات الأصلية بشكل خطير.
                        لا يُنصح باستخدام هذه البيانات في بيئات الإنتاج. جرب:
                        - زيادة عدد الـ Epochs
                        - استخدام طريقة CTGAN
                        - زيادة حجم البيانات الأصلية
                        """)

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء معالجة الملف: {e}")
            st.info("💡 تأكد أن الملف يحتوي على بيانات صحيحة وأعمدة غير فارغة تماماً.")

with tab2:
    st.markdown("""
    ### 🧬 حول محرك التوليد — SOL Platform
    
    تم بناء هذا النظام لتلبية المتطلبات التالية:
    
    ---
    
    #### طرق التوليد المدعومة:
    
    | الطريقة | الوصف | المميزات | العيوب |
    |---------|-------|----------|--------|
    | **📊 إحصائية (Basic)** | تولد كل عمود بشكل مستقل بنفس التوزيع والنسب | سريعة جداً (مع Threads) | لا تحافظ على العلاقات بين الأعمدة |
    | **🧠 CTGAN (ذكاء اصطناعي)** | شبكة عصبية توليدية تتعلم العلاقات المخفية | تحافظ على الارتباطات والعلاقات المنطقية | أبطأ، تحتاج بيانات كافية |
    
    ---
    
    #### المميزات:
    - **⚡ Threads (تعدد المسارات)**: يتم توليد الأعمدة الآن بالتوازي مما يسرع العملية بشكل كبير.
    - **💾 Session State**: لن تفقد بياناتك المولدة عند الضغط على أزرار التحميل.
    - **📁 أسماء ملفات ديناميكية**: الملفات تأخذ اسم الملف الأصلي تلقائياً (مثال: SOL-ires_synthetic.csv).
    - **🤖 مساعد الذكاء الاصطناعي (LLM)**: تحليل هيكل بياناتك وتقديم توصيات بـ Google Gemini.
    - **التحليل الإحصائي**: يتم دراسة التوزيع والارتباطات بين كل الأعمدة.
    - **البيانات الحساسة**: يتم اكتشاف واستبدال بيانات المستخدمين (أسماء، إيميلات، هواتف) ببيانات وهمية باستخدام `Faker`.
    - **مقارنة مرئية وتقرير (Fidelity Score)**: لضمان مستوى دقة البيانات الناتجة وصلاحيتها لتدريب نماذج الذكاء الاصطناعي.
    - **تصدير شامل**: تحميل البيانات بصيغة CSV أو تقرير Excel متعدد الصفحات.
    - **مصفوفة فرق الارتباطات**: لقياس مدى حفاظ البيانات المولدة على العلاقات الأصلية.
    - **🔒 تحليل الخصوصية (DCR)**: يقيس المسافة بين كل صف مولد وأقرب صف حقيقي لضمان عدم تسريب بيانات المستخدمين الأصليين، متوافق مع معايير GDPR.
    - **⚖️ التوليد الموجه (Class Balancing)**: القدرة على إجبار خوارزمية التوليد على تغيير التوزيع الأصلي لبعض الأعمدة الفئوية لمساواتها (لحل مشكلة انحياز البيانات Data Bias).
    - **🦠 مختبر الضوضاء (Noise Lab)**: إمكانية تلويث البيانات المولدة بالقيم المفقودة أو الشاذة لاختبار جودة النماذج في بيئات قاسية.
    - **🛡️ الشروط المخصصة (Custom Constraints)**: إمكانية كتابة قواعد برمجية لاستبعاد البيانات التي لا تتطابق مع منطق الأعمال.
    - **🌐 النشر الفوري (Mock API)**: تحويل البيانات المولدة إلى خادم FastAPI حي بضغطة زر واحدة.
    """)
