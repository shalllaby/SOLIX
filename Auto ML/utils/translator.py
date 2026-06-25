import streamlit as st

# Translation Catalog for bilingual interface (En/Ar)
TRANSLATIONS = {
    "en": {
        "title": "🤖 SOL AutoML System",
        "subtitle": "Production-Grade Intelligent AutoML Engine — From raw dataset to robust edge deployments",
        "sidebar_nav": "Navigation Hub",
        "mode_selection": "Workspace Mode",
        "mode_auto": "⚡ Automatic AI Mode",
        "mode_expert": "🛠️ Manual Expert Mode",
        "lang_selection": "Language / اللغة",
        "dataset_upload": "1. Upload Clean Dataset",
        "upload_help": "Supports CSV, XLSX, and JSON datasets",
        "browse_files": "Browse Files",
        "no_data": "Please upload a dataset to begin the analysis.",
        "dataset_preview": "Dataset Preview & Analytics",
        "data_stats": "Dataset Statistics",
        "rows": "Rows",
        "cols": "Columns",
        "memory": "Memory Usage",
        "column_profile": "Column Data Profiles",
        "target_selection": "2. Target Column & ML Task Confirmation",
        "auto_detect_target": "🔍 Enable Auto Detect Target",
        "target_dropdown": "Select Target Column",
        "target_explanation": "AutoML Target Candidate Ranking:",
        "confirm_target_btn": "Confirm Target Column",
        "target_confirmed_msg": "Target column confirmed: **{}**",
        "task_detected_msg": "Inferred ML Task: **{}**",
        "task_binary": "Binary Classification",
        "task_multiclass": "Multiclass Classification",
        "task_regression": "Regression",
        
        # Advanced settings in Manual mode
        "expert_settings": "Expert Hyperparameter Workspace",
        "train_split": "Train/Test Split Ratio (%)",
        "scaling_opt": "Feature Scaling Strategy",
        "encoding_opt": "Categorical Encoding Strategy",
        "cv_folds": "Cross Validation Folds",
        "model_selection_title": "Select Models to Train",
        "tuning_toggle": "Enable Hyperparameter Tuning (Lightweight RandomSearchCV)",
        "tune_folds_lbl": "Tuning Folds",
        "timeout_lbl": "Max Training Duration (Seconds)",
        
        # Action Buttons
        "start_training": "🚀 Initialize SOL AutoML Parallel Pipeline",
        "cancel_training": "🛑 Cancel Execution",
        "training_active": "Training models concurrently... Please wait.",
        "training_cancelled_msg": "Training was cancelled by the user.",
        "training_timeout_msg": "Training timed out after exceeding the maximum duration limit.",
        "training_success_msg": "SOL AutoML Concurrent Pipeline completed successfully!",
        
        # Results
        "results_dashboard": "🏆 SOL Intelligent Performance Dashboard",
        "leaderboard_tbl": "Model Generalization Leaderboard",
        "feature_importance_plot": "Feature Influence Profile (Plotly)",
        "confusion_matrix_plot": "Confusion Matrix Analysis (Plotly)",
        "residual_plot": "Residuals Distribution (Plotly)",
        "actual_vs_pred": "Actual vs. Predicted Scatter (Plotly)",
        
        # Optimization
        "deep_opt_btn": "💎 Execute Stage 2: Deep Hyperparameter Optimization",
        "deep_opt_desc": "Performs quick randomized grid sweeps on the champion model to maximize score.",
        "deep_opt_success": "Deep optimization complete! Best tuned model metrics updated.",
        
        # Exports
        "downloads_title": "📦 Download & Export Production Artifacts",
        "dl_zip_btn": "Download SOL Complete Production Bundle (.ZIP)",
        "dl_pdf_btn": "Download Executive PDF Performance Report",
        "pdf_lang_lbl": "Select Report Language",
        
        # UI Pills & Alerts
        "p_numerical": "Numerical",
        "p_categorical": "Categorical",
        "p_datetime": "Datetime",
        "no_importance": "Feature importance is not available or could not be estimated for the best model.",
        
        # SOL New Bilingual Terms
        "composite_score_lbl": "Composite Score",
        "health_status_lbl": "Health Status",
        "generalization_gap_lbl": "Gen Gap",
        "cv_mean_lbl": "CV Mean",
        "cv_std_lbl": "CV Std",
        "stability_lbl": "Stability",
        "val_metric_lbl": "Val Score",
        "train_metric_lbl": "Train Score",
        "runtime_lbl": "Fit Time",
        "prediction_playground_title": "🎮 Live Interactive Inference Playground",
        "prediction_playground_desc": "Test predictions interactively. Input features below to run predictions against the champion model.",
        "predict_btn": "Execute Model Inference",
        "predict_outcome_lbl": "Model Predicted Target Output:",
        "heavy_dataset_alert": "⚠️ Heavy Dataset Detected! Automatically optimizing settings (Reducing CV folds and disabling slow models) to prevent memory crashes."
    },
    
    "ar": {
        "title": "🤖 نظام SOL AutoML المطور",
        "subtitle": "منصة التعلم الآلي الذكي المتكاملة — من البيانات الخام إلى تشغيل نماذج الإنتاج الفعلية المستقرة",
        "sidebar_nav": "مركز التحكم والخيارات",
        "mode_selection": "وضع مساحة العمل",
        "mode_auto": "⚡ وضع الذكاء الاصطناعي التلقائي",
        "mode_expert": "🛠️ وضع الخبير اليدوي",
        "lang_selection": "Language / اللغة",
        "dataset_upload": "1. رفع مجموعة البيانات النظيفة",
        "upload_help": "يدعم صيغ CSV و XLSX و JSON",
        "browse_files": "تصفح الملفات",
        "no_data": "يرجى رفع مجموعة بيانات لبدء التحليل.",
        "dataset_preview": "معاينة وتحليلات مجموعة البيانات",
        "data_stats": "إحصائيات مجموعة البيانات",
        "rows": "الصفوف",
        "cols": "الأعمدة",
        "memory": "استهلاك الذاكرة",
        "column_profile": "الملفات التعريفية للأعمدة",
        "target_selection": "2. تأكيد العمود المستهدف ومهمة التعلم الآلي",
        "auto_detect_target": "🔍 تفعيل الكشف التلقائي عن الهدف",
        "target_dropdown": "اختر العمود المستهدف (Target)",
        "target_explanation": "ترتيب أعمدة الهدف المقترحة ذكياً:",
        "confirm_target_btn": "تأكيد العمود المستهدف",
        "target_confirmed_msg": "تم تأكيد العمود المستهدف: **{}**",
        "task_detected_msg": "مهمة التعلم الآلي المستنتجة: **{}**",
        "task_binary": "تصنيف ثنائي (Binary Classification)",
        "task_multiclass": "تصنيف متعدد الفئات (Multiclass Classification)",
        "task_regression": "انحدار وتوقع قيم مستمرة (Regression)",
        
        # Advanced settings in Manual mode
        "expert_settings": "إعدادات الخبراء المتقدمة",
        "train_split": "نسبة تقسيم بيانات التدريب / الاختبار (%)",
        "scaling_opt": "استراتيجية تقييس الخصائص (Scaling)",
        "encoding_opt": "استراتيجية ترميز المتغيرات الفئوية (Encoding)",
        "cv_folds": "عدد طيات التحقق المتقاطع (CV Folds)",
        "model_selection_title": "اختر النماذج المطلوب تدريبها",
        "tuning_toggle": "تفعيل ضبط المعاملات الفائقة (RandomSearchCV خفيف)",
        "tune_folds_lbl": "طيات التحقق للضبط",
        "timeout_lbl": "الحد الأقصى لوقت التدريب (بالثواني)",
        
        # Action Buttons
        "start_training": "🚀 تشغيل وتدريب نماذج التعلم الآلي المتوازية",
        "cancel_training": "🛑 إلغاء عملية التدريب",
        "training_active": "جاري تدريب النماذج بالتوازي... يرجى الانتظار.",
        "training_cancelled_msg": "تم إلغاء عملية التدريب بواسطة المستخدم.",
        "training_timeout_msg": "انتهى الوقت المحدد للتدريب قبل اكتمال جميع النماذج.",
        "training_success_msg": "اكتملت عملية تدريب واستخلاص نماذج SOL بنجاح!",
        
        # Results
        "results_dashboard": "🏆 لوحة مؤشرات الأداء والتحليلات التفاعلية الذكية",
        "leaderboard_tbl": "جدول متصدري أداء النماذج المختبرة",
        "feature_importance_plot": "الملف التعريفي لتأثير وأهمية الأعمدة (Plotly)",
        "confusion_matrix_plot": "تحليل مصفوفة الارتباك (Plotly)",
        "residual_plot": "توزيع البواقي والأخطاء (Plotly)",
        "actual_vs_pred": "مخطط التشتت للقيم الفعلية مقابل المتوقعة (Plotly)",
        
        # Optimization
        "deep_opt_btn": "💎 تشغيل المرحلة 2: الضبط العميق للمعاملات الفائقة",
        "deep_opt_desc": "يقوم بعمليات فحص شبكية عشوائية سريعة على النموذج البطل لرفع دقة وجودة التوقعات.",
        "deep_opt_success": "اكتمل الضبط العميق بنجاح! تم تحديث مقاييس أداء النموذج البطل بعد التحسين.",
        
        # Exports
        "downloads_title": "📦 تحميل وتصدير حزم التشغيل والإنتاج",
        "dl_zip_btn": "تحميل حزمة ملفات إنتاج SOL الكاملة (.ZIP)",
        "dl_pdf_btn": "تحميل التقرير التنفيذي المعتمد بصيغة PDF",
        "pdf_lang_lbl": "اختر لغة التقرير",
        
        # UI Pills & Alerts
        "p_numerical": "رقمي",
        "p_categorical": "فئوي",
        "p_datetime": "زمني",
        "no_importance": "أهمية الأعمدة غير مدعومة أو لم يتم استخلاصها للنموذج الأفضل حالياً.",
        
        # SOL New Bilingual Terms
        "composite_score_lbl": "التقييم المركب",
        "health_status_lbl": "الحالة الصحية للنموذج",
        "generalization_gap_lbl": "فجوة التعميم",
        "cv_mean_lbl": "متوسط CV",
        "cv_std_lbl": "انحراف CV",
        "stability_lbl": "مؤشر الاستقرار",
        "val_metric_lbl": "درجة التحقق",
        "train_metric_lbl": "درجة التدريب",
        "runtime_lbl": "وقت التدريب",
        "prediction_playground_title": "🎮 منصة اختبار وتوقعات النموذج التفاعلية فورياً",
        "prediction_playground_desc": "قم باختبار جودة توقعات النموذج. أدخل قيم الأعمدة أدناه لإرسال التوقع الفوري إلى النموذج الفائز.",
        "predict_btn": "بدء حساب وتوقع النموذج",
        "predict_outcome_lbl": "النتيجة المتوقعة المستخلصة من النموذج الفائز:",
        "heavy_dataset_alert": "⚠️ تم الكشف عن مجموعة بيانات ضخمة! تم تحسين الإعدادات تلقائياً (تقليل طيات التحقق وإلغاء النماذج البطيئة) لتفادي أي مشاكل في الذاكرة."
    }
}

def inject_rtl_styles(is_arabic: bool):
    """
    Injects dynamic premium CSS layouts.
    If Arabic is selected, activates global RTL direction wrappers, 
    card alignments, correct margins, and smooth typography.
    Also injects a cohesive dark-metallic theme interface.
    """
    # 1. Base Premium styling for both modes
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Cairo:wght@400;600;700&display=swap');
    
    /* Main body typography & premium dark metallic styling */
    .stApp {
        font-family: 'Cairo', sans-serif !important;
        background: radial-gradient(circle at 50% 50%, #0F172A 0%, #020617 100%) !important;
        color: #F8FAFC !important;
    }
    
    /* Metric Card Premium Design */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #3B82F6, #10B981) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        margin-bottom: 12px !important;
    }
    
    /* Premium Headers */
    .gradient-header {
        font-family: 'Cairo', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #60A5FA 0%, #34D399 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 15px !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # 2. Dynamic RTL injector for Arabic Mode
    if is_arabic:
        rtl_css = """
        <style>
        /* Force Right to Left direction */
        html, body, [data-testid="stAppViewContainer"], .main, .sidebar {
            direction: RTL !important;
            text-align: right !important;
            font-family: 'Cairo', sans-serif !important;
        }
        
        /* Adjust alignment for key Streamlit components */
        div[data-testid="stMarkdownContainer"] p, 
        div[data-testid="stMarkdownContainer"] h1, 
        div[data-testid="stMarkdownContainer"] h2, 
        div[data-testid="stMarkdownContainer"] h3, 
        div[data-testid="stMarkdownContainer"] h4, 
        div[data-testid="stMarkdownContainer"] h5, 
        div[data-testid="stMarkdownContainer"] h6 {
            text-align: right !important;
            direction: RTL !important;
        }
        
        /* Form inputs, selectboxes, multi-select direction */
        .stSelectbox, .stMultiSelect, .stSlider, .stTextInput, .stNumberInput {
            direction: RTL !important;
            text-align: right !important;
        }
        
        /* Align Metric containers right */
        div[data-testid="stMetric"] {
            text-align: right !important;
            direction: RTL !important;
        }
        
        /* Fix checkboxes and radio buttons alignment */
        div[data-testid="stCheckbox"], div[data-testid="stRadio"] {
            direction: RTL !important;
            text-align: right !important;
        }
        
        /* Sidebar layout adjustments */
        [data-testid="stSidebar"] {
            direction: RTL !important;
            text-align: right !important;
        }
        
        /* Flexbox grid elements alignment */
        div[data-testid="column"] {
            direction: RTL !important;
            text-align: right !important;
        }
        </style>
        """
        st.markdown(rtl_css, unsafe_allow_html=True)
