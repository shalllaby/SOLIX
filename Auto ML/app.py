import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import gc
import os
import traceback
import threading
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from core.kaggle_client import KaggleWorkflowManager

logger = logging.getLogger("SOL.KaggleMLOps")

# Import core modules
from core.analyzer import analyze_dataset, rank_target_candidates, infer_task_type
from core.preprocessor import prepare_data, get_processed_feature_names, sanitize_features
from core.llm_profiler import resolve_api_key, run_profiler, get_category_label, get_confidence_icon
from core.llm_triage import get_llm_model_triage
from core.engine import AutoMLTrainingEngine, AutoMLValidationError
from core.exporter import AutoMLArtifactExporter
from core.visualizer import SOLAutoMLVisualizer
from utils.pdf_generator import AutoMLPDFReportGenerator
from utils.translator import TRANSLATIONS, inject_rtl_styles

# Page Config
st.set_page_config(
    page_title="SOL AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Initialize Session States
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "workspace_mode" not in st.session_state:
    st.session_state.workspace_mode = "auto"
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "profile" not in st.session_state:
    st.session_state.profile = None
if "target_confirmed" not in st.session_state:
    st.session_state.target_confirmed = False
if "confirmed_target" not in st.session_state:
    st.session_state.confirmed_target = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None

# Training Pipeline Caches
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "training_cancelled" not in st.session_state:
    st.session_state.training_cancelled = False
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = None
if "trained_instances" not in st.session_state:
    st.session_state.trained_instances = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "target_encoder" not in st.session_state:
    st.session_state.target_encoder = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "stage2_complete" not in st.session_state:
    st.session_state.stage2_complete = False

# AI Agentic Profiler Caches
if "ai_profiler_result" not in st.session_state:
    st.session_state.ai_profiler_result = None
if "ai_blacklist" not in st.session_state:
    st.session_state.ai_blacklist = []
if "model_status" not in st.session_state:
    st.session_state.model_status = {}
if "model_errors" not in st.session_state:
    st.session_state.model_errors = {}
if "training_logs_buffer" not in st.session_state:
    st.session_state.training_logs_buffer = ""
if "timings" not in st.session_state:
    st.session_state.timings = {}

# Kaggle Training Caches
if "kaggle_active" not in st.session_state:
    st.session_state.kaggle_active = False
if "kaggle_status" not in st.session_state:
    st.session_state.kaggle_status = ""
if "kaggle_error" not in st.session_state:
    st.session_state.kaggle_error = None
if "kaggle_kernel_ref" not in st.session_state:
    st.session_state.kaggle_kernel_ref = None
if "training_backend" not in st.session_state:
    st.session_state.training_backend = "local"

# Sliced validation dataset caches
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None


# 2. Setup Localization Utilities
lang = st.session_state.lang
t = TRANSLATIONS[lang]
is_arabic = (lang == "ar")

# Inject Dark Metallic LTR/RTL Styles
inject_rtl_styles(is_arabic)


# 3. Sidebar Navigation Panel
st.sidebar.markdown(f"<h2 class='gradient-header'>{t['sidebar_nav']}</h2>", unsafe_allow_html=True)

# Language Toggle
selected_lang = st.sidebar.selectbox(
    t["lang_selection"],
    options=["English", "العربية"],
    index=0 if st.session_state.lang == "en" else 1
)
new_lang = "en" if selected_lang == "English" else "ar"
if new_lang != st.session_state.lang:
    st.session_state.lang = new_lang
    st.rerun()

# Workspace Mode Select
workspace_mode = st.sidebar.radio(
    t["mode_selection"],
    options=["auto", "expert"],
    format_func=lambda x: t["mode_auto"] if x == "auto" else t["mode_expert"]
)
if workspace_mode != st.session_state.workspace_mode:
    st.session_state.workspace_mode = workspace_mode
    # Reset target selections to prevent pollution
    st.session_state.target_confirmed = False
    st.session_state.confirmed_target = None
    st.session_state.leaderboard = None
    st.session_state.best_model = None
    st.session_state.stage2_complete = False
    st.rerun()

# Global Timeout Protection Slider
timeout_limit = st.sidebar.slider(
    t["timeout_lbl"],
    min_value=30,
    max_value=1800,
    value=300,
    step=30
)

# Groq API Key Input (Secure Password Field)
st.sidebar.markdown("---")
groq_sidebar_label = "🧠 Groq API Key" if not is_arabic else "🧠 مفتاح Groq API"
st.sidebar.text_input(
    groq_sidebar_label,
    type="password",
    key="groq_api_key_input",
    help="Enter your Groq API key to enable the AI Agentic Profiler. Get one at console.groq.com" if not is_arabic else "أدخل مفتاح Groq API لتفعيل المحلل الذكي. احصل على مفتاح من console.groq.com"
)

st.sidebar.markdown("---")
st.sidebar.caption("SOL AutoML Platform v2.0 • Stable Release")


# 4. Main Panel Layout
st.markdown(f"<h1 class='gradient-header'>{t['title']}</h1>", unsafe_allow_html=True)
st.write(t["subtitle"])
st.markdown("---")

# Section 1: File Uploader
st.markdown(f"### {t['dataset_upload']}")
uploaded_file = st.file_uploader(
    t["upload_help"],
    type=["csv", "xlsx", "json"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    if st.session_state.dataset_name != uploaded_file.name:
        try:
            name = uploaded_file.name
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
                
            # Full column sanitization BEFORE training for LightGBM safety
            df = sanitize_features(df)
            
            st.session_state.dataset = df
            st.session_state.dataset_name = name
            st.session_state.profile = analyze_dataset(df)
            st.session_state.target_confirmed = False
            st.session_state.confirmed_target = None
            st.session_state.leaderboard = None
            st.session_state.best_model = None
            st.session_state.stage2_complete = False
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
else:
    st.session_state.dataset = None
    st.session_state.dataset_name = ""
    st.session_state.profile = None
    st.session_state.target_confirmed = False
    st.session_state.confirmed_target = None
    st.session_state.leaderboard = None
    st.session_state.best_model = None
    st.session_state.stage2_complete = False

# Stop execution if no dataset uploaded
if st.session_state.dataset is None:
    st.info(t["no_data"])
    st.stop()


# 5. Dataset Preview & Profiling Dashboard
df = st.session_state.dataset
profile = st.session_state.profile

# Heavy Dataset Size Protection
memory_mb = profile['memory_usage_bytes'] / (1024 * 1024)
is_heavy_dataset = (memory_mb > 15.0 or df.shape[0] * df.shape[1] > 150_000)

with st.expander(t["dataset_preview"], expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><strong>{t['rows']}:</strong> {profile['shape'][0]}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><strong>{t['cols']}:</strong> {profile['shape'][1]}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><strong>{t['memory']}:</strong> {memory_mb:.2f} MB</div>", unsafe_allow_html=True)
        
    st.dataframe(df.head(5), width="stretch")
    
    # Detail Column Profile
    st.markdown(f"#### {t['column_profile']}")
    details_data = []
    for col, meta in profile["column_details"].items():
        mapped_type = meta["type"]
        if "Numerical" in mapped_type:
            pill = f"🟢 {t['p_numerical']}"
        elif "Categorical" in mapped_type:
            pill = f"🔵 {t['p_categorical']}"
        else:
            pill = f"📅 {t['p_datetime']}"
            
        details_data.append({
            "Column": col,
            "Detected Type": pill,
            "Missing Count": meta["missing_count"],
            "Missing %": f"{meta['missing_pct']:.1f}%",
            "Unique Values": meta["unique_count"],
            "Sample Values": str(meta["sample_values"][:3])
        })
    st.dataframe(pd.DataFrame(details_data), width="stretch")

st.markdown("---")


# 5.5 AI Agentic Data Profiler (Powered by Groq)
profiler_hdr = "🧠 AI Agentic Data Profiler" if not is_arabic else "🧠 المحلل الذكي للبيانات"
st.markdown(f"### {profiler_hdr}")

groq_key = resolve_api_key()
ai_result = st.session_state.ai_profiler_result

if not groq_key:
    st.info(
        "💡 Enter your Groq API Key in the sidebar to enable AI-powered dataset analysis "
        "(target suggestion, leakage detection, domain identification)."
        if not is_arabic else
        "💡 أدخل مفتاح Groq API في الشريط الجانبي لتفعيل التحليل الذكي للبيانات "
        "(اقتراح العمود المستهدف، كشف تسرب البيانات، تحديد المجال)."
    )
else:
    analyze_btn_label = "🔍 Analyze Dataset with AI" if not is_arabic else "🔍 تحليل البيانات بالذكاء الاصطناعي"
    if st.button(analyze_btn_label, type="secondary", key="ai_profiler_trigger") or ai_result is not None:
        if ai_result is None:
            with st.spinner("🧠 SOL is analyzing your dataset semantically via Groq..." if not is_arabic else "🧠 يقوم SOL بتحليل بياناتك دلالياً عبر Groq..."):
                ai_result = run_profiler(df, groq_key)
                st.session_state.ai_profiler_result = ai_result

        if ai_result is not None:
            meta = ai_result.get("_meta", {})
            latency = meta.get("latency_ms", "?")
            st.caption(f"⚡ Groq Response: {latency}ms | Model: {meta.get('model', 'N/A')} | Tokens: {meta.get('input_tokens', 0)} in / {meta.get('output_tokens', 0)} out")

            # Domain & Summary
            domain_label = "🌌 Domain Detected" if not is_arabic else "🌌 المجال المكتشف"
            st.markdown(f"**{domain_label}:** {ai_result['domain_detected']}")
            st.markdown(f"📝 {ai_result['summary']}")

            st.markdown("---")

            # Target Suggestion
            target_info = ai_result["suggested_target"]
            conf_icon = get_confidence_icon(target_info["confidence"])
            target_box_hdr = "🎯 AI-Suggested Target Column" if not is_arabic else "🎯 العمود المستهدف المقترح من الذكاء الاصطناعي"
            st.markdown(f"#### {target_box_hdr}")
            st.markdown(
                f"> **`{target_info['column_name']}`** &nbsp; {conf_icon} "
                f"Confidence: **{target_info['confidence'].upper()}**\n>"
                f"\n> *{target_info['reasoning']}*"
            )

            # Blacklist
            blacklist_entries = ai_result.get("blacklist", [])
            if blacklist_entries:
                bl_hdr = f"🚫 Dynamic Blacklist ({len(blacklist_entries)} columns flagged)" if not is_arabic else f"🚫 القائمة السوداء الديناميكية ({len(blacklist_entries)} أعمدة مشبوهة)"
                st.markdown(f"#### {bl_hdr}")

                checked_blacklist = []
                for entry in blacklist_entries:
                    cat_label = get_category_label(entry["reason_category"], is_arabic)
                    checkbox_label = f"`{entry['column_name']}` — {cat_label}"
                    is_checked = st.checkbox(
                        checkbox_label,
                        value=True,
                        key=f"bl_{entry['column_name']}",
                        help=entry["reasoning"]
                    )
                    if is_checked:
                        checked_blacklist.append(entry["column_name"])
                    st.caption(f"    ↳ {entry['reasoning']}")

                st.session_state.ai_blacklist = checked_blacklist
            else:
                no_bl_msg = "✅ No data leakage or noise columns detected." if not is_arabic else "✅ لم يتم اكتشاف أعمدة تسرب بيانات أو ضوضاء."
                st.success(no_bl_msg)
                st.session_state.ai_blacklist = []
        else:
            st.warning(
                "⚠️ AI Profiler could not complete analysis. Falling back to statistical profiler."
                if not is_arabic else
                "⚠️ تعذر على المحلل الذكي إكمال التحليل. سيتم استخدام المحلل الإحصائي."
            )

st.markdown("---")


# 6. Target Selection Panel (Interactive Confirmation)
st.markdown(f"### {t['target_selection']}")

# Pre-fill from AI suggestion if available
ai_suggested_target = None
if ai_result and ai_result.get("suggested_target", {}).get("column_name") in df.columns:
    ai_suggested_target = ai_result["suggested_target"]["column_name"]

auto_detect = st.checkbox(t["auto_detect_target"], value=True)
ranked_targets = rank_target_candidates(df, profile["column_types"])

if auto_detect:
    st.markdown(f"##### {t['target_explanation']}")
    for idx, cand in enumerate(ranked_targets[:3]):
        icon = "🏆" if idx == 0 else "🥈" if idx == 1 else "🥉"
        reason = cand["reason_ar"] if is_arabic else cand["reason_en"]
        ai_tag = " 🧠" if (ai_suggested_target and cand["column"] == ai_suggested_target) else ""
        st.markdown(f"> **{icon} Candidate {idx+1}: `{cand['column']}`{ai_tag}** (Confidence Score: {cand['score']})\n> *Reason: {reason}*")

    target_options = [c["column"] for c in ranked_targets]
    # If AI suggested a target, move it to position 0
    if ai_suggested_target and ai_suggested_target in target_options:
        target_options.remove(ai_suggested_target)
        target_options.insert(0, ai_suggested_target)
else:
    target_options = list(df.columns)
    if ai_suggested_target and ai_suggested_target in target_options:
        target_options.remove(ai_suggested_target)
        target_options.insert(0, ai_suggested_target)

selected_target = st.selectbox(
    t["target_dropdown"],
    options=target_options,
    index=0 if (auto_detect and ranked_targets) else 0
)

col_btn1, col_btn2 = st.columns([2, 5])
with col_btn1:
    confirm_btn = st.button(t["confirm_target_btn"], type="primary")

if confirm_btn or st.session_state.target_confirmed:
    st.session_state.target_confirmed = True
    st.session_state.confirmed_target = selected_target

    inferred_task = infer_task_type(df, selected_target, profile["column_types"])

    # OVERRIDE GUARD: If target is string-typed, force classification regardless of inference
    if df[selected_target].dtype == "object":
        if df[selected_target].nunique() == 2:
            inferred_task = "binary"
        else:
            inferred_task = "multiclass"

    # If AI profiler provided a task_type and user hasn't explicitly overridden, prefer it
    ai_task = ai_result.get("task_type") if ai_result else None

    # Manual Task Type Override UI
    override_label = "🎯 Task Type Override" if not is_arabic else "🎯 تجاوز نوع المهمة يدوياً"
    task_override = st.radio(
        override_label,
        options=["auto", "classification", "regression"],
        format_func=lambda x: {
            "auto": (f"🤖 Auto-Detect ({inferred_task.capitalize()})" + (f" | AI: {ai_task}" if ai_task else "")) if not is_arabic else (f"🤖 اكتشاف تلقائي ({inferred_task})" + (f" | ذكاء: {ai_task}" if ai_task else "")),
            "classification": "📊 Force Classification (Binary/Multiclass)" if not is_arabic else "📊 فرض التصنيف (ثنائي/متعدد الفئات)",
            "regression": "📈 Force Regression (Continuous)" if not is_arabic else "📈 فرض الانحدار (متغير مستمر)"
        }[x],
        index=0,
        horizontal=True,
        key="task_type_override_radio"
    )

    if task_override == "classification":
        if df[selected_target].nunique() == 2:
            final_task = "binary"
        else:
            final_task = "multiclass"
        if inferred_task == "regression":
            st.warning("⚠️ Override Active: Auto-detection suggested Regression, but you forced Classification." if not is_arabic else "⚠️ تجاوز نشط: اقترح الاكتشاف التلقائي الانحدار، لكنك فرضت التصنيف.")
    elif task_override == "regression":
        final_task = "regression"
        if inferred_task in ["binary", "multiclass"]:
            st.warning("⚠️ Override Active: Auto-detection suggested Classification, but you forced Regression." if not is_arabic else "⚠️ تجاوز نشط: اقترح الاكتشاف التلقائي التصنيف، لكنك فرضت الانحدار.")
    else:
        final_task = inferred_task

    st.session_state.task_type = final_task

    # Model Routing / LLM Triage step (Agentic Triage)
    if "llm_triage_result" not in st.session_state or st.session_state.get("llm_triage_target") != selected_target or st.session_state.get("llm_triage_task") != final_task:
        api_key = resolve_api_key()
        triage_res = get_llm_model_triage(df, selected_target, final_task, api_key)
        if triage_res:
            st.session_state.llm_triage_result = triage_res
            st.session_state.llm_triage_target = selected_target
            st.session_state.llm_triage_task = final_task
            
            # Automatically pre-fill checkbox state in st.session_state
            for m in triage_res.get("approved_models", []):
                st.session_state[f"chk_{m}"] = True
            for m in triage_res.get("rejected_models", []):
                st.session_state[f"chk_{m}"] = False

    task_label = t["task_binary"] if final_task == "binary" else t["task_multiclass"] if final_task == "multiclass" else t["task_regression"]

    st.success(t["target_confirmed_msg"].format(selected_target))
    st.info(t["task_detected_msg"].format(task_label))

    # Display Triage Report
    if "llm_triage_result" in st.session_state:
        triage_res = st.session_state.llm_triage_result
        approved = triage_res.get("approved_models", [])
        rejected = triage_res.get("rejected_models", [])
        reasoning = triage_res.get("reasoning", {})
        
        triage_title = "🤖 SOL Model Routing & Triage Report" if not is_arabic else "🤖 تقرير SOL لتوجيه وتصنيف النماذج"
        with st.expander(triage_title, expanded=True):
            st.markdown("##### Approved Models / النماذج المعتمدة")
            cols_app = st.columns(len(approved) if approved else 1)
            for idx, m in enumerate(approved):
                with cols_app[idx % len(cols_app)]:
                    st.success(f"✔️ {m}")
                    
            if rejected:
                st.markdown("##### Rejected Models / النماذج المستبعدة")
                cols_rej = st.columns(len(rejected) if rejected else 1)
                for idx, m in enumerate(rejected):
                    with cols_rej[idx % len(cols_rej)]:
                        st.error(f"❌ {m}")
                        
            st.markdown("---")
            st.markdown("##### Model Reasoning & Justification / المبررات الفنية لاختيار واستبعاد النماذج")
            for m, reason in reasoning.items():
                icon = "🟢" if m in approved else "🔴"
                st.markdown(f"**{icon} {m}**: {reason}")

    # Display active blacklist summary
    active_bl = st.session_state.get("ai_blacklist", [])
    if active_bl:
        bl_summary = ", ".join([f"`{c}`" for c in active_bl])
        st.warning(f"🚫 AI Blacklist Active: {bl_summary} will be excluded from training." if not is_arabic else f"🚫 القائمة السوداء نشطة: {bl_summary} سيتم استبعادها من التدريب.")

if not st.session_state.target_confirmed:
    st.stop()

st.markdown("---")


# 7. Model Training Workspace (Automatic vs Manual Settings)
is_expert = (st.session_state.workspace_mode == "expert")

scaling_method = "standard"
encoding_method = "onehot"
test_size = 0.2
selected_models = []
enable_tuning = False
cv_folds = 5

if is_heavy_dataset:
    st.warning(t["heavy_dataset_alert"])
    cv_folds = 3  # Force lower folds to prevent OOM
    
if is_expert:
    st.markdown(f"### {t['expert_settings']}")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        test_pct = st.slider(t["train_split"], min_value=10, max_value=50, value=20, step=5)
        test_size = test_pct / 100.0
        
        scaling_method = st.selectbox(
            t["scaling_opt"],
            options=["standard", "minmax", "robust", "none"]
        )
        
        encoding_method = st.selectbox(
            t["encoding_opt"],
            options=["onehot", "ordinal", "none"]
        )
        
    with col_exp2:
        cv_folds = st.selectbox(t["cv_folds"], options=[3, 5, 10], index=0 if is_heavy_dataset else 1)
        enable_tuning = st.checkbox(t["tuning_toggle"], value=False)
        
    dummy_engine = AutoMLTrainingEngine(st.session_state.task_type)
    all_supported_models = dummy_engine.get_available_models()
    
    st.markdown(f"##### {t['model_selection_title']}")
    col_chk1, col_chk2 = st.columns(2)
    with col_chk1:
        for m in all_supported_models[:len(all_supported_models)//2]:
            default_val = True
            if "llm_triage_result" in st.session_state:
                default_val = m in st.session_state.llm_triage_result.get("approved_models", [])
            if st.checkbox(m, value=default_val, key=f"chk_{m}"):
                selected_models.append(m)
    with col_chk2:
        for m in all_supported_models[len(all_supported_models)//2:]:
            is_heavy_svm = (is_heavy_dataset and m == "SVM")
            default_val = not is_heavy_svm
            if "llm_triage_result" in st.session_state:
                default_val = m in st.session_state.llm_triage_result.get("approved_models", [])
            if st.checkbox(m, value=default_val, key=f"chk_{m}"):
                selected_models.append(m)
else:
    # Automatic Mode defaults
    test_size = 0.2
    scaling_method = "standard"
    encoding_method = "onehot"
    enable_tuning = False
    
    num_features = [col for col in profile["column_types"]["numerical"] if col != st.session_state.confirmed_target]
    cat_features = [col for col in profile["column_types"]["categorical"] if col != st.session_state.confirmed_target]
    shape = (df.shape[0], len(num_features) + len(cat_features))
    
    dummy_engine = AutoMLTrainingEngine(st.session_state.task_type)
    if "llm_triage_result" in st.session_state:
        selected_models = st.session_state.llm_triage_result.get("approved_models", [])
    else:
        selected_models = dummy_engine.select_smart_models(shape)
# 7.5 Compute Backend Configuration
st.markdown("---")
backend_hdr = "☁️ Compute Backend Configuration" if not is_arabic else "☁️ تكوين بيئة التشغيل والحوسبة"
st.markdown(f"### {backend_hdr}")

training_backend = st.radio(
    "Choose Compute Infrastructure / اختر بيئة التدريب الحوسبية:",
    options=["local", "kaggle"],
    format_func=lambda x: ("💻 Local Host Processing (Uses local CPU/RAM)" if not is_arabic else "💻 معالجة محلية (تستهلك موارد المعالج والذاكرة)") if x == "local" else ("⚡ Kaggle Cloud Environment (Zero local resource consumption)" if not is_arabic else "⚡ بيئة Kaggle السحابية (استهلاك صفري لموارد الجهاز المحلي)"),
    key="training_backend_radio"
)
st.session_state.training_backend = training_backend

kaggle_user = ""
kaggle_token = ""
if training_backend == "kaggle":
    st.info("Kaggle Cloud Mode: Training runs in a secure remote Kaggle sandbox. Output artifacts will be automatically downloaded to your local host upon completion.")
    
    default_username = ""
    default_key = ""
    try:
        if "kaggle" in st.secrets:
            default_username = st.secrets["kaggle"].get("KAGGLE_USERNAME", "")
            default_key = st.secrets["kaggle"].get("KAGGLE_API_TOKEN", "")
    except Exception:
        pass
        
    if not default_key:
        default_key = "KGAT_0034d6fd413ada3d3b57d06d1736d0ae"
    if not default_username:
        default_username = "al_dalil_governance_service"
        
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        kaggle_user = st.text_input("Kaggle Username", value=default_username)
    with col_k2:
        kaggle_token = st.text_input("Kaggle API Key/Token", value=default_key, type="password")

# 8. AutoML Pipeline Execution Space
train_trigger = False
if st.session_state.kaggle_active:
    st.markdown("### ☁️ Kaggle Cloud Training Status")
    st.info(f"🔄 **Current Action**: {st.session_state.kaggle_status}")
    st.spinner("Executing remotely... Local CPU/RAM usage is at 0%.")
    if st.session_state.kaggle_kernel_ref:
        st.caption(f"Kaggle Kernel: `{st.session_state.kaggle_kernel_ref}`")
    time.sleep(5)
    st.rerun()
elif st.session_state.kaggle_error:
    st.markdown("### ☁️ Kaggle Cloud Training Failed")
    st.error(f"❌ Error details: {st.session_state.kaggle_error}")
    if st.button("Clear Error / Reset State"):
        st.session_state.kaggle_error = None
        st.rerun()
else:
    col_act1, col_act2 = st.columns([2, 5])
    with col_act1:
        train_trigger = st.button(t["start_training"], type="primary")

if train_trigger:
    st.session_state.training_active = True
    st.session_state.training_cancelled = False
    st.session_state.leaderboard = None
    st.session_state.best_model = None
    st.session_state.stage2_complete = False
    st.session_state.training_logs_buffer = ""
    st.session_state.timings = {}
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_dashboard = st.empty()
    console_logs = st.empty()
    
    logs_output = []
    model_states = {m: "pending" for m in selected_models}
    
    def on_progress(model_name: str, progress: float, info: Dict[str, Any]):
        progress_bar.progress(progress)
        status_val = info["status"]
        if status_val == "success":
            status_val = "completed"
        elif status_val == "training":
            status_val = "running"
            
        model_states[model_name] = status_val
        
        status_label = t["training_active"]
        if status_val == "running":
            msg = f"⏱️ Training Model {model_name}..." if not is_arabic else f"⏱️ جاري تدريب النموذج {model_name}..."
            logs_output.append(msg)
        elif status_val == "completed":
            msg = f"✅ Model {model_name} completed in {info['metrics']['fit_time']:.2f}s." if not is_arabic else f"✅ اكتمل النموذج {model_name} في {info['metrics']['fit_time']:.2f} ثواني."
            logs_output.append(msg)
        else:
            err = info.get('error', 'Unknown Error')
            if status_val == "timeout":
                msg = f"⏳ Model {model_name} training timed out." if not is_arabic else f"⏳ انتهى وقت تدريب النموذج {model_name}."
            elif status_val == "skipped":
                msg = f"🚫 Model {model_name} was skipped." if not is_arabic else f"🚫 تم تخطي النموذج {model_name}."
            elif status_val == "incompatible dataset":
                msg = f"⚠️ Model {model_name} is incompatible with this dataset." if not is_arabic else f"⚠️ النموذج {model_name} غير متوافق مع مجموعة البيانات."
            else:
                msg = f"❌ Model {model_name} failed: {err}" if not is_arabic else f"❌ فشل النموذج {model_name}: {err}"
            logs_output.append(msg)
            
        status_text.text(f"{status_label} ({int(progress*100)}%)")
        console_logs.markdown("```text\n" + "\n".join(logs_output[-6:]) + "\n```")
        
        labels_dict = {
            "completed": "Completed ✅" if not is_arabic else "مكتمل ✅",
            "failed": "Failed ❌" if not is_arabic else "فشل ❌",
            "timeout": "Timeout ⏳" if not is_arabic else "انتهى الوقت ⏳",
            "skipped": "Skipped 🚫" if not is_arabic else "تم تخطيه 🚫",
            "incompatible dataset": "Incompatible Dataset ⚠️" if not is_arabic else "غير متوافق مع البيانات ⚠️",
            "running": "Running 🔄" if not is_arabic else "جاري التشغيل 🔄",
            "pending": "Pending 💤" if not is_arabic else "بالانتظار 💤",
        }
        
        db_title = "SOL AutoML Model Execution Dashboard" if not is_arabic else "لوحة تدريب نماذج التعلم الآلي"
        dashboard_content = f"### 🦾 {db_title}\n"
        for m, st_val in model_states.items():
            lbl = labels_dict.get(st_val, st_val)
            dashboard_content += f"- **{m}**: {lbl}\n"
            
        status_dashboard.markdown(dashboard_content)

    t_start = time.time()
    try:
        # Preprocess features
        X_train_p, X_test_p, y_train, y_test, preprocessor, target_encoder = prepare_data(
            df=df,
            target_col=st.session_state.confirmed_target,
            col_types=profile["column_types"],
            task_type=st.session_state.task_type,
            test_size=test_size,
            scaling_method=scaling_method,
            encoding_method=encoding_method,
            blacklist=st.session_state.get("ai_blacklist", [])
        )
        
        # Cache splits
        st.session_state.X_train = X_train_p
        st.session_state.X_test = X_test_p
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.preprocessor = preprocessor
        st.session_state.target_encoder = target_encoder
        
        if st.session_state.training_backend == "kaggle":
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            
            run_id = int(time.time())
            dataset_slug = f"dataset-automl-{run_id}"
            kernel_slug = f"kernel-automl-{run_id}"
            
            # Start background thread
            st.session_state.kaggle_active = True
            st.session_state.kaggle_status = "Initializing remote environment..."
            st.session_state.kaggle_error = None
            st.session_state.kaggle_kernel_ref = None
            st.session_state.leaderboard = None
            st.session_state.best_model = None
            st.session_state.trained_instances = None
            st.session_state.model_status = {}
            st.session_state.model_errors = {}
            
            def remote_thread():
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                add_script_run_ctx(threading.current_thread(), ctx)
                logger.info("Asynchronous remote training background thread initialized.")
                try:
                    logger.info("Instantiating KaggleWorkflowManager...")
                    manager = KaggleWorkflowManager(kaggle_user, kaggle_token)
                    
                    logger.info("Uploading preprocessed splits to Kaggle...")
                    st.session_state.kaggle_status = "Uploading preprocessed splits to Kaggle secure cloud..."
                    dataset_ref = manager.upload_preprocessed_splits(
                        X_train=X_train_p,
                        y_train=y_train,
                        X_test=X_test_p,
                        y_test=y_test,
                        dataset_slug=dataset_slug,
                        title=f"AutoML Splits {run_id}"
                    )
                    
                    logger.info("Dataset successfully uploaded as: '%s'", dataset_ref)
                    st.session_state.kaggle_status = "Triggering remote Kaggle notebook execution..."
                    kernel_ref = manager.trigger_kernel(
                        dataset_ref=dataset_ref,
                        kernel_slug=kernel_slug,
                        task_type=st.session_state.task_type,
                        models=selected_models
                    )
                    st.session_state.kaggle_kernel_ref = kernel_ref
                    logger.info("Kaggle notebook triggered successfully. Ref: '%s'", kernel_ref)
                    
                    attempt = 0
                    while True:
                        attempt += 1
                        status = manager.get_status(kernel_ref)
                        logger.info("Polling remote run status (Attempt %d): %s", attempt, status.upper())
                        
                        if status == "complete":
                            logger.info("Remote AutoML training run completed successfully. Initializing output download...")
                            st.session_state.kaggle_status = "Downloading remote training outputs..."
                            dest_dir = Path("downloads")
                            manager.download_outputs(kernel_ref, dest_dir)
                            
                            import joblib
                            best_model_path = dest_dir / "best_model.pkl"
                            metrics_path = dest_dir / "metrics.json"
                            
                            if best_model_path.exists() and metrics_path.exists():
                                logger.info("Loading Champion Model '%s' and metrics into local memory...", best_model_path)
                                st.session_state.best_model = joblib.load(best_model_path)
                                with open(metrics_path, "r") as f:
                                    metrics_data = json.load(f)
                                
                                if isinstance(metrics_data, dict) and "leaderboard" in metrics_data:
                                    leaderboard = metrics_data["leaderboard"]
                                    st.session_state.leaderboard = leaderboard
                                    best_name = leaderboard[0]["model_name"] if leaderboard else None
                                    st.session_state.best_model_name = best_name
                                    st.session_state.trained_instances = {
                                        best_name: st.session_state.best_model
                                    } if best_name else {}
                                    st.session_state.model_status = metrics_data.get("model_status", {})
                                    st.session_state.model_errors = metrics_data.get("model_errors", {})
                                else:
                                    st.session_state.best_model_name = metrics_data.get("model_name")
                                    st.session_state.leaderboard = [metrics_data]
                                    st.session_state.trained_instances = {
                                        metrics_data.get("model_name"): st.session_state.best_model
                                    }
                                    st.session_state.model_status = {metrics_data.get("model_name"): "completed"}
                                
                                st.session_state.training_logs_buffer = "Remote Kaggle run completed successfully."
                                st.session_state.stage2_complete = True
                                
                                # Extract Feature Importance using unified engine method
                                feature_names = get_processed_feature_names(preprocessor)
                                engine_helper = AutoMLTrainingEngine(task_type=st.session_state.task_type)
                                st.session_state.feature_importance = engine_helper.extract_feature_importance(
                                    model=st.session_state.best_model,
                                    feature_names=feature_names,
                                    X_val=X_test_p,
                                    y_val=y_test
                                )
                                
                                logger.info("Remote champion model loaded successfully. Ready to run local inference.")
                                st.session_state.kaggle_status = "Completed"
                                st.session_state.kaggle_active = False
                                break
                            else:
                                raise FileNotFoundError("Outputs downloaded but best_model.pkl/metrics.json not found.")
                        elif status == "error":
                            raise RuntimeError("Kaggle Kernel execution failed with error state.")
                        else:
                            st.session_state.kaggle_status = f"Running remotely (status: {status})...."
                        time.sleep(10)
                except Exception as ex:
                    logger.error("Remote Kaggle execution failed with exception: %s", str(ex))
                    st.session_state.kaggle_error = str(ex)
                    st.session_state.kaggle_status = "Failed"
                    st.session_state.kaggle_active = False
            
            thread = threading.Thread(target=remote_thread, daemon=True)
            thread.start()
            
            progress_bar.empty()
            status_text.empty()
            console_logs.empty()
            status_dashboard.empty()
            st.session_state.training_active = False
            st.rerun()
            
        else:
            # Initialize parallel engine
            engine = AutoMLTrainingEngine(
                task_type=st.session_state.task_type,
                timeout_seconds=timeout_limit,
                model_timeout_seconds=60.0,
                progress_callback=on_progress
            )
            
            leaderboard, trained_instances = engine.train_baselines(
                X_train=X_train_p,
                y_train=y_train,
                X_test=X_test_p,
                y_test=y_test,
                model_names=selected_models,
                cv_folds=cv_folds
            )
            
            # Save session states
            st.session_state.model_status = engine.model_status
            st.session_state.model_errors = engine.model_errors
            st.session_state.training_logs_buffer = "\n".join(logs_output)
            st.session_state.timings = {"baseline_fit_duration": time.time() - t_start}
            
            progress_bar.empty()
            status_text.empty()
            console_logs.empty()
            status_dashboard.empty()
            
            st.session_state.leaderboard = leaderboard
            st.session_state.trained_instances = trained_instances
        
        if not leaderboard:
            st.warning("⚠️ No compatible models could be trained on this dataset.")
            st.session_state.best_model_name = None
            st.session_state.best_model = None
            st.session_state.feature_importance = None
        else:
            best_meta = leaderboard[0]
            st.session_state.best_model_name = best_meta["model_name"]
            st.session_state.best_model = trained_instances[best_meta["model_name"]]
            
            # Extract Feature Importance
            feature_names = get_processed_feature_names(preprocessor)
            st.session_state.feature_importance = engine.extract_feature_importance(
                model=st.session_state.best_model,
                feature_names=feature_names,
                X_val=X_test_p,
                y_val=y_test
            )
            st.success(t["training_success_msg"])
            
        st.session_state.training_active = False
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        console_logs.empty()
        status_dashboard.empty()
        st.error(f"Execution Error: {e}")
        st.code(traceback.format_exc())
        st.session_state.training_active = False


# 9. Results Panel & Insights (Leaderboard + Visualizations)
if st.session_state.leaderboard is not None:
    st.markdown(f"### {t['results_dashboard']}")
    
    # Persistent Execution Status Dashboard
    st.markdown("#### 🦾 " + ("Execution & Model Status" if not is_arabic else "حالة تشغيل النماذج الفنية"))
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        st.markdown("**Successful Models / النماذج الناجحة:**")
        for m_name, m_status in st.session_state.model_status.items():
            if m_status == "completed":
                st.write(f"✅ **{m_name}**")
    with col_st2:
        st.markdown("**Failed, Skipped, or Timed-out Models / النماذج المتعثرة والمستبعدة:**")
        non_success_exist = False
        for m_name, m_status in st.session_state.model_status.items():
            if m_status != "completed":
                non_success_exist = True
                icon = "❌" if m_status == "failed" else ("⏳" if m_status == "timeout" else ("🚫" if m_status == "skipped" else "⚠️"))
                st.write(f"{icon} **{m_name}** ({m_status})")
        if not non_success_exist:
            st.write("✨ None / لا يوجد")
            
    # Technical logs expander
    if st.session_state.model_errors:
        with st.expander("🔍 View Technical Debugging Logs / سجل الأخطاء التفصيلي للنماذج"):
            for m_name, err_trace in st.session_state.model_errors.items():
                st.markdown(f"##### 🟥 **{m_name}** Error Log:")
                st.code(err_trace, language="text")
                
    if not st.session_state.leaderboard:
        st.warning("⚠️ No compatible models could be trained on this dataset." if not is_arabic else "⚠️ لم يتمكن النظام من تدريب أي نموذج متوافق مع مجموعة البيانات هذه.")
    else:
        st.markdown("---")
        
        # Leaderboard Table
        st.markdown(f"##### {t['leaderboard_tbl']}")
        
        # Format Leaderboard columns beautiful and localized
        lb_display_data = []
        for r in st.session_state.leaderboard:
            metric_key = "f1" if st.session_state.task_type in ["binary", "multiclass"] else "r2"
            val_metric = r["val_metrics"].get(metric_key, 0.0)
            train_metric = r["train_metrics"].get(metric_key, 0.0)
            
            # Map custom bilingual labels
            labels_str = " | ".join(r["labels"])
            
            lb_display_data.append({
                t["leaderboard_tbl"] if not is_arabic else "اسم النموذج": r["model_name"],
                t["composite_score_lbl"]: r["composite_score"],
                t["val_metric_lbl"]: val_metric,
                t["train_metric_lbl"]: train_metric,
                t["cv_mean_lbl"]: r["cv_mean"],
                t["cv_std_lbl"]: r["cv_std"],
                t["generalization_gap_lbl"]: r["generalization_gap"],
                t["stability_lbl"]: max(0.0, 1.0 - 5.0 * r["cv_std"]),
                t["health_status_lbl"]: labels_str,
                t["runtime_lbl"]: r["fit_time"]
            })
            
        leaderboard_df = pd.DataFrame(lb_display_data)
        st.dataframe(leaderboard_df, width="stretch")
        
        # Champion Display Card
        champion_name = st.session_state.best_model_name
        score_metric = "F1-Score" if st.session_state.task_type in ["binary", "multiclass"] else "R2-Score"
        best_score = st.session_state.leaderboard[0]["composite_score"]
        
        st.info(f"🏆 **Champion Model:** `{champion_name}` | SOL Composite Balanced score: **{best_score:.6f}**")
        
        # Stage 2: Deep Optimization Workspace
        st.markdown("---")
        st.markdown(f"#### 💎 Stage 2: Deep Optimization")
        st.write(t["deep_opt_desc"])
        
        col_opt1, col_opt2 = st.columns([2, 5])
        with col_opt1:
            is_kaggle_run = (st.session_state.training_backend == "kaggle")
            deep_opt_trigger = st.button(t["deep_opt_btn"], disabled=st.session_state.stage2_complete or is_kaggle_run)
            if is_kaggle_run:
                st.caption("Deep Optimization is handled automatically in the remote Kaggle cloud pipeline.")
            
        if deep_opt_trigger:
            with st.spinner("Tuning hyperparameters... Please wait."):
                t0_opt = time.time()
                engine = AutoMLTrainingEngine(st.session_state.task_type)
                tuned_model, tuned_metrics = engine.deep_optimize_best_model(
                    X_train=st.session_state.X_train,
                    y_train=st.session_state.y_train,
                    X_test=st.session_state.X_test,
                    y_test=st.session_state.y_test,
                    best_model_name=st.session_state.best_model_name,
                    best_model=st.session_state.best_model,
                    cv_folds=cv_folds
                )
                st.session_state.best_model = tuned_model
                st.session_state.timings["deep_opt_duration"] = time.time() - t0_opt
                
                # Replace metric in leaderboard cache
                for idx, r in enumerate(st.session_state.leaderboard):
                    if r["model_name"] == st.session_state.best_model_name:
                        tuned_metrics["model_name"] = st.session_state.best_model_name
                        st.session_state.leaderboard[idx] = tuned_metrics
                        break
                        
                # Re-extract feature importance
                feature_names = get_processed_feature_names(st.session_state.preprocessor)
                st.session_state.feature_importance = engine.extract_feature_importance(
                    model=tuned_model,
                    feature_names=feature_names,
                    X_val=st.session_state.X_test,
                    y_val=st.session_state.y_test
                )
                
                st.session_state.stage2_complete = True
                st.success(t["deep_opt_success"])
                st.rerun()
                
        st.markdown("---")
        
        # 10. Generate Visualizations (Interactive Plotly + Safe Static PNG exports)
        st.markdown(f"#### {('Visualizations & Insights' if not is_arabic else 'التحليلات والمخططات التفاعلية')}")
        
        # Pre-build dictionary for static zip images in background
        visualizations_png = {}
        
        col_g1, col_g2 = st.columns(2)
        
        model = st.session_state.best_model
        X_val = st.session_state.X_test
        y_val = st.session_state.y_test
        preds = model.predict(X_val)
        
        # Load classes for classification
        labels = []
        is_cls = (st.session_state.task_type in ["binary", "multiclass"])
        if is_cls:
            unique_classes = np.unique(y_val)
            labels = [str(c) for c in unique_classes]
            if st.session_state.target_encoder is not None:
                labels = [str(l) for l in st.session_state.target_encoder.classes_]
        
        # Chart 1: Feature Importance
        with col_g1:
            if st.session_state.feature_importance:
                fig_fi = SOLAutoMLVisualizer.plot_feature_importance(st.session_state.feature_importance)
                st.plotly_chart(fig_fi, width="stretch")
                
                # Double-fallback background PNG generation for export
                SOLAutoMLVisualizer.save_figure(
                    fig_fi,
                    "temp_visuals/feature_importance.png",
                    SOLAutoMLVisualizer._fallback_feature_importance,
                    (st.session_state.feature_importance,)
                )
                if os.path.exists("temp_visuals/feature_importance.png"):
                    with open("temp_visuals/feature_importance.png", "rb") as f:
                        visualizations_png["feature_importance"] = f.read()
            else:
                st.warning(t["no_importance"])
                
        # Chart 2: Task Specific Visual
        with col_g2:
            if is_cls:
                # Confusion Matrix
                fig_cm = SOLAutoMLVisualizer.plot_confusion_matrix(y_val, preds, labels)
                st.plotly_chart(fig_cm, width="stretch")
                
                SOLAutoMLVisualizer.save_figure(
                    fig_cm,
                    "temp_visuals/confusion_matrix.png",
                    SOLAutoMLVisualizer._fallback_confusion_matrix,
                    (y_val, preds, labels)
                )
                if os.path.exists("temp_visuals/confusion_matrix.png"):
                    with open("temp_visuals/confusion_matrix.png", "rb") as f:
                        visualizations_png["confusion_matrix"] = f.read()
            else:
                # Actual vs Predicted
                fig_ap = SOLAutoMLVisualizer.plot_pred_vs_actual(y_val, preds)
                st.plotly_chart(fig_ap, width="stretch")
                
                SOLAutoMLVisualizer.save_figure(
                    fig_ap,
                    "temp_visuals/pred_vs_actual.png",
                    SOLAutoMLVisualizer._fallback_pred_vs_actual,
                    (y_val, preds)
                )
                if os.path.exists("temp_visuals/pred_vs_actual.png"):
                    with open("temp_visuals/pred_vs_actual.png", "rb") as f:
                        visualizations_png["pred_vs_actual"] = f.read()
                        
        col_g3, col_g4 = st.columns(2)
        
        # Chart 3: Additional Task plots
        with col_g3:
            if is_cls:
                # ROC Curve
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val)
                    fig_roc = SOLAutoMLVisualizer.plot_roc_curve(y_val, probs, st.session_state.task_type)
                    st.plotly_chart(fig_roc, width="stretch")
                    
                    SOLAutoMLVisualizer.save_figure(
                        fig_roc,
                        "temp_visuals/roc_curve.png",
                        SOLAutoMLVisualizer._fallback_roc_curve,
                        (y_val, probs, st.session_state.task_type)
                    )
                    if os.path.exists("temp_visuals/roc_curve.png"):
                        with open("temp_visuals/roc_curve.png", "rb") as f:
                            visualizations_png["roc_curve"] = f.read()
            else:
                # Residuals plot
                fig_res = SOLAutoMLVisualizer.plot_residual_plot(y_val, preds)
                st.plotly_chart(fig_res, width="stretch")
                
                SOLAutoMLVisualizer.save_figure(
                    fig_res,
                    "temp_visuals/residual_plot.png",
                    SOLAutoMLVisualizer._fallback_residual_plot,
                    (y_val, preds)
                )
                if os.path.exists("temp_visuals/residual_plot.png"):
                    with open("temp_visuals/residual_plot.png", "rb") as f:
                        visualizations_png["residual_plot"] = f.read()
                        
        # Chart 4: Leaderboard Comparer & Runtime Comparer
        with col_g4:
            base_metric = "f1" if is_cls else "r2"
            fig_lb = SOLAutoMLVisualizer.plot_leaderboard_comparison(pd.DataFrame(st.session_state.leaderboard), base_metric)
            st.plotly_chart(fig_lb, width="stretch")
            
            SOLAutoMLVisualizer.save_figure(
                fig_lb,
                "temp_visuals/leaderboard_comparison.png",
                SOLAutoMLVisualizer._fallback_leaderboard_comparison,
                (pd.DataFrame(st.session_state.leaderboard), base_metric)
            )
            if os.path.exists("temp_visuals/leaderboard_comparison.png"):
                with open("temp_visuals/leaderboard_comparison.png", "rb") as f:
                    visualizations_png["leaderboard_comparison"] = f.read()
                    
        # Clean up temp_visuals directory safely to prevent memory leak
        import shutil
        try:
            shutil.rmtree("temp_visuals", ignore_errors=True)
        except Exception:
            pass
            
        st.markdown("---")
        
        
        # 11. Interactive Inference Prediction Playground UI
        st.markdown(f"### {t['prediction_playground_title']}")
        st.write(t["prediction_playground_desc"])
        
        # Dynamically inspect prep bundle structure columns
        preprocessor = st.session_state.preprocessor
        numerical_features = [col for col in profile["column_types"]["numerical"] if col != st.session_state.confirmed_target]
        categorical_features = [col for col in profile["column_types"]["categorical"] if col != st.session_state.confirmed_target]
        
        # Grid input fields
        inputs_dict = {}
        
        st.markdown("##### " + ("Provide Inference Features / أدخل قيم الخصائص الفنية:" if not is_arabic else "أدخل قيم الخصائص الفنية المطلوبة:"))
        col_in1, col_in2 = st.columns(2)
        
        # Numerical Inputs
        with col_in1:
            for feat in numerical_features:
                col_min = float(df[feat].min()) if not df[feat].isna().all() else 0.0
                col_max = float(df[feat].max()) if not df[feat].isna().all() else 100.0
                col_med = float(df[feat].median()) if not df[feat].isna().all() else 0.0
                
                inputs_dict[feat] = st.number_input(
                    label=f"🟢 {feat} (Range: {col_min:.2f} - {col_max:.2f})",
                    min_value=col_min,
                    max_value=col_max,
                    value=col_med,
                    key=f"inf_{feat}"
                )
                
        # Categorical Inputs
        with col_in2:
            for feat in categorical_features:
                unique_vals = list(df[feat].dropna().unique())
                if not unique_vals:
                    unique_vals = ["missing"]
                inputs_dict[feat] = st.selectbox(
                    label=f"🔵 {feat}",
                    options=unique_vals,
                    key=f"inf_{feat}"
                )
                
        # Trigger Inference Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        col_inf_btn, _ = st.columns([2, 5])
        with col_inf_btn:
            predict_btn = st.button(t["predict_btn"], type="primary")
            
        if predict_btn:
            try:
                # Convert inputs into single row Pandas DataFrame
                inference_row = pd.DataFrame([inputs_dict])
                
                # Sanitize features strictly matching training rules
                inference_row = sanitize_features(inference_row)
                
                # Transform features safely
                inf_processed = preprocessor.transform(inference_row)
                
                # Format processed inputs back into aligned dataframe to avoid mismatch
                try:
                    feature_names = get_processed_feature_names(preprocessor)
                    sanitized_features = []
                    seen = {}
                    for col in feature_names:
                        san = sanitize_features(pd.DataFrame(columns=[col])).columns[0]
                        if san in seen:
                            seen[san] += 1
                            san_unique = f"{san}_{seen[san]}"
                        else:
                            seen[san] = 0
                            san_unique = san
                        sanitized_features.append(san_unique)
                        
                    if isinstance(inf_processed, pd.DataFrame):
                        inf_processed.columns = sanitized_features
                    else:
                        inf_processed = pd.DataFrame(inf_processed, columns=sanitized_features)
                except Exception:
                    if not isinstance(inf_processed, pd.DataFrame):
                        inf_processed = pd.DataFrame(inf_processed)
                        
                # Reorder / Align strictly to champion fitted names
                expected_features = None
                if hasattr(model, "feature_names_in_"):
                    expected_features = list(model.feature_names_in_)
                elif hasattr(model, "feature_name"):
                    try:
                        expected_features = list(model.feature_name())
                    except Exception:
                        pass
                elif hasattr(model, "get_booster"):
                    try:
                        expected_features = model.get_booster().feature_names
                    except Exception:
                        pass
                        
                if expected_features:
                    inf_processed = inf_processed[expected_features]
                    
                # Predict Outcome!
                raw_pred = model.predict(inf_processed)[0]
                
                # Decode outcomes
                if is_cls and st.session_state.target_encoder is not None:
                    decoded_pred = st.session_state.target_encoder.inverse_transform([raw_pred])[0]
                else:
                    decoded_pred = raw_pred
                    
                # Render beautifully in a big alert box
                st.markdown("---")
                st.markdown(f"#### 🎯 {t['predict_outcome_lbl']}")
                st.markdown(f"<div style='background: rgba(16, 185, 129, 0.1); border: 2px solid #10B981; border-radius: 12px; padding: 20px; text-align: center; font-size: 2.2rem; font-weight: 700; color: #10B981;'>{decoded_pred}</div>", unsafe_allow_html=True)
                
            except Exception as inf_err:
                st.error(f"Inference prediction failed: {inf_err}")
                st.code(traceback.format_exc())
                
        st.markdown("---")
        
        
        # 12. Downloads & Executive Report Exporter Space
        st.markdown(f"### {t['downloads_title']}")
        
        col_dl1, col_dl2 = st.columns(2)
        
        # 1. Executive PDF report generation
        pdf_stream = None
        try:
            pdf_stream = AutoMLPDFReportGenerator.generate_report(
                dataset_name=st.session_state.dataset_name,
                task_type=st.session_state.task_type,
                target_col=st.session_state.confirmed_target,
                metrics={
                    "dataset_rows": profile["shape"][0],
                    "dataset_cols": profile["shape"][1]
                },
                col_types=profile["column_types"],
                best_model_name=st.session_state.best_model_name,
                leaderboard=st.session_state.leaderboard,
                feature_importance=st.session_state.feature_importance if st.session_state.feature_importance else [],
                visualizations_dict=visualizations_png,
                is_arabic=is_arabic
            )
        except Exception as pdf_err:
            st.warning(f"Warning: PDF Executive Report could not be compiled. Continuing export bundle... ({pdf_err})")
            # Build simple plain fallbacks
            pdf_stream = io.BytesIO(b"Plain text executive placeholder due to ReportLab drawing exception.")
            
        # 2. Package ZIP Bundle
        zip_stream = None
        try:
            zip_stream = AutoMLArtifactExporter.serialize_to_zip(
                best_model=st.session_state.best_model,
                preprocessor=st.session_state.preprocessor,
                target_encoder=st.session_state.target_encoder,
                metrics=st.session_state.leaderboard[0],
                col_types=profile["column_types"],
                task_type=st.session_state.task_type,
                target_col=st.session_state.confirmed_target,
                dataset_shape=profile["shape"],
                best_model_name=st.session_state.best_model_name,
                feature_importance=st.session_state.feature_importance if st.session_state.feature_importance else [],
                original_df=df,
                pdf_report_bytes=pdf_stream.getvalue(),
                visualizations_dict=visualizations_png,
                failed_models_log=st.session_state.model_errors,
                training_logs=st.session_state.training_logs_buffer,
                timings_dict=st.session_state.timings
            )
        except Exception as zip_err:
            st.error(f"Failed to compile SOL ZIP bundle: {zip_err}")
            
        with col_dl1:
            if zip_stream:
                st.download_button(
                    label=t["dl_zip_btn"],
                    data=zip_stream,
                    file_name=f"SOL AutoML - {st.session_state.dataset_name}.zip",
                    mime="application/zip",
                    width="stretch"
                )
                
        with col_dl2:
            if pdf_stream:
                st.download_button(
                    label=t["dl_pdf_btn"],
                    data=pdf_stream,
                    file_name=f"SOL_AutoML_Executive_Report_{st.session_state.confirmed_target}.pdf",
                    mime="application/pdf",
                    width="stretch"
                )
