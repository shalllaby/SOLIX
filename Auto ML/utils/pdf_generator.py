import os
import io
import time
import tempfile
from typing import Dict, List, Any, Optional
from PIL import Image as PILImage

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Reshaping and BiDi support for Arabic PDF
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_PDF_READY = True
except ImportError:
    ARABIC_PDF_READY = False


class AutoMLPDFReportGenerator:
    """
    Upgraded executive PDF Report Generator for SOL AutoML.
    Compiles beautiful, multilingual (Arabic/English) executive summaries,
    scoring matrices, overfit analysis, and embeds static visualization PNGs.
    """
    
    @staticmethod
    def _register_fonts() -> bool:
        """
        Attempts to register standard Windows Arial font for high-quality multilingual support.
        Falls back to standard Helvetica if Arial is unavailable.
        """
        # Standard paths on Windows
        paths = [
            (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
            (r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf"),
            (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf")
        ]
        
        for regular, bold in paths:
            if os.path.exists(regular):
                try:
                    pdfmetrics.registerFont(TTFont("MultilingualArial", regular))
                    if os.path.exists(bold):
                        pdfmetrics.registerFont(TTFont("MultilingualArial-Bold", bold))
                    return True
                except Exception:
                    pass
        return False

    @staticmethod
    def _process_text(text: str, is_arabic: bool, font_registered: bool) -> str:
        if not text:
            return ""
        if is_arabic and ARABIC_PDF_READY and font_registered:
            try:
                reshaped = arabic_reshaper.reshape(text)
                return get_display(reshaped)
            except Exception:
                pass
        return text

    @classmethod
    def generate_report(
        cls,
        dataset_name: str,
        task_type: str,
        target_col: str,
        metrics: Dict[str, Any],
        col_types: Dict[str, List[str]],
        best_model_name: str,
        leaderboard: List[Dict[str, Any]],
        feature_importance: List[Dict[str, Any]],
        visualizations_dict: Dict[str, bytes],  # key -> png bytes
        is_arabic: bool = False
    ) -> io.BytesIO:
        # Coerce inputs safely to string
        dataset_name = str(dataset_name) if dataset_name else "dataset"
        task_type = str(task_type) if task_type else "classification"
        target_col = str(target_col) if target_col else "target"
        best_model_name = str(best_model_name) if best_model_name else "model"

        # Coerce feature_importance if passed as a dictionary
        if isinstance(feature_importance, dict):
            feature_importance = [
                {"feature": f, "importance": float(i)}
                for f, i in feature_importance.items()
            ]
        elif not feature_importance:
            feature_importance = []

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        font_registered = cls._register_fonts()
        font_name = "MultilingualArial" if font_registered else "Helvetica"
        bold_font_name = "MultilingualArial-Bold" if font_registered else "Helvetica-Bold"
        
        styles = getSampleStyleSheet()
        
        # Style overrides matching modern premium palette
        title_style = ParagraphStyle(
            name="ReportTitle",
            fontName=bold_font_name,
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1E3A8A"), # Deep Blue
            alignment=2 if (is_arabic and font_registered) else 0,
            spaceAfter=15
        )
        
        h1_style = ParagraphStyle(
            name="SectionHeading",
            fontName=bold_font_name,
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2C4A7F"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
            alignment=2 if (is_arabic and font_registered) else 0
        )
        
        body_style = ParagraphStyle(
            name="BodyTextCustom",
            fontName=font_name,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#374151"),
            spaceAfter=6,
            alignment=2 if (is_arabic and font_registered) else 0
        )
        
        bold_body_style = ParagraphStyle(
            name="BoldBodyTextCustom",
            fontName=bold_font_name,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
            alignment=2 if (is_arabic and font_registered) else 0
        )
        
        story = []
        
        t = {
            "title": "SOL AutoML Executive Performance Report" if not is_arabic else "نظام SOL AutoML - التقرير التنفيذي لمؤشرات الأداء",
            "subtitle": f"Dataset: {dataset_name} | Date: {time.strftime('%Y-%m-%d %H:%M:%S')}" if not is_arabic else f"قاعدة البيانات: {dataset_name} | التاريخ: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "overview_title": "1. Dataset Profiling & Overview" if not is_arabic else "1. الملف التعريفي والتحليلي لقاعدة البيانات",
            "overview_desc": "Intelligent dataset schema identification and target analysis completed successfully." if not is_arabic else "تم اكتمال الكشف الذكي لملف الأعمدة وتحليل الهدف بنجاح.",
            "rows": "Total Dataset Rows" if not is_arabic else "إجمالي صفوف البيانات",
            "cols": "Total Dataset Columns" if not is_arabic else "إجمالي الأعمدة",
            "target_col": "Selected Target Column" if not is_arabic else "العمود المستهدف المختار",
            "task_type": "Inferred ML Task" if not is_arabic else "مهمة التعلم الآلي المستنتجة",
            "numerical_cnt": "Numerical Features" if not is_arabic else "الأعمدة الرقمية",
            "categorical_cnt": "Categorical Features" if not is_arabic else "الأعمدة الفئوية",
            
            "leaderboard_title": "2. Comprehensive Model Leaderboard & Health Evaluation" if not is_arabic else "2. لوحة متصدري أداء النماذج وتشخيصات السلامة",
            "leaderboard_desc": "Sorted by SOL Composite Score. Overfitting, stability, variance, and complexity are fully integrated." if not is_arabic else "مرتبة حسب درجة التقييم المركبة (SOL Composite Score) مع دمج الفروق العامة والاستقرار ونسب التضخم.",
            
            "best_model_title": "3. Champion Model Deep Performance Analysis" if not is_arabic else "3. التحليلات التفصيلية للنموذج البطل الفائز",
            "best_model_desc": f"The selected champion algorithm is: {best_model_name}." if not is_arabic else f"النموذج البطل الفائز بأعلى تقييم متوازن هو: {best_model_name}.",
            
            "importance_title": "4. Relative Feature Influence Profiles" if not is_arabic else "4. الهيكل النسبي لتأثير وأهمية الأعمدة",
            "recommend_title": "5. Production Deployment Instructions" if not is_arabic else "5. إرشادات وخطوات التشغيل والإنتاج الفعلي",
            "rec_1": "1. Deploy 'trained_model.pkl' and 'preprocessing_pipeline.pkl' into your production environment." if not is_arabic else "1. قم بحفظ وتجهيز ملفات 'trained_model.pkl' و 'preprocessing_pipeline.pkl' في بيئة التشغيل المستهدفة.",
            "rec_2": "2. Use 'predict.py' as your standalone deployment framework for zero-configuration setup." if not is_arabic else "2. استخدم ملف 'predict.py' كإطار تشغيل برمي لسرعة التوقع دون إعدادات معقدة.",
            "rec_3": "3. Preprocess inference payloads strictly through the loaded preprocessor before feed predictions." if not is_arabic else "3. قم بتهيئة مدخلات التوقعات الفورية برمجياً باستخدام preprocessor المرفق قبل التمرير للنموذج.",
            "rec_4": "4. Setup shadow deployments and shadow tests initially to verify stability under heavy workloads." if not is_arabic else "4. قم بإعداد نشر موازٍ متبوع باختبارات صامتة مبدئياً للتحقق من سلامة الأداء واستقرار التوقعات."
        }
        
        # Header
        story.append(Paragraph(cls._process_text(t["title"], is_arabic, font_registered), title_style))
        story.append(Paragraph(cls._process_text(t["subtitle"], is_arabic, font_registered), body_style))
        story.append(Spacer(1, 10))
        
        # Section 1: Overview
        story.append(Paragraph(cls._process_text(t["overview_title"], is_arabic, font_registered), h1_style))
        story.append(Paragraph(cls._process_text(t["overview_desc"], is_arabic, font_registered), body_style))
        
        num_feats = len(col_types.get("numerical", []))
        cat_feats = len(col_types.get("categorical", []))
        
        metadata_data = [
            [cls._process_text(t["rows"], is_arabic, font_registered), f"{metrics.get('dataset_rows', 'N/A')}"],
            [cls._process_text(t["cols"], is_arabic, font_registered), f"{metrics.get('dataset_cols', 'N/A')}"],
            [cls._process_text(t["target_col"], is_arabic, font_registered), f"'{target_col}'"],
            [cls._process_text(t["task_type"], is_arabic, font_registered), f"{task_type.capitalize()}"],
            [cls._process_text(t["numerical_cnt"], is_arabic, font_registered), f"{num_feats}"],
            [cls._process_text(t["categorical_cnt"], is_arabic, font_registered), f"{cat_feats}"]
        ]
        
        meta_table = Table(metadata_data, colWidths=[200, 320])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#F9FAFB")),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#374151")),
            ('FONTNAME', (0,0), (-1,-1), font_name),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F3F4F6")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB")),
            ('ALIGN', (0,0), (-1,-1), 'RIGHT' if is_arabic else 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))
        
        # Section 2: Leaderboard
        story.append(Paragraph(cls._process_text(t["leaderboard_title"], is_arabic, font_registered), h1_style))
        story.append(Paragraph(cls._process_text(t["leaderboard_desc"], is_arabic, font_registered), body_style))
        
        is_cls = (task_type in ["binary", "multiclass"])
        metric_lbl = "F1-Score" if is_cls else "R2-Score"
        metric_key = "f1" if is_cls else "r2"
        
        headers = [
            "Model", "Val " + metric_lbl, "CV Mean", "CV Std", "Gen Gap", "Composite", "Health Status"
        ] if not is_arabic else [
            "النموذج", "أداء " + metric_lbl, "متوسط CV", "انحراف CV", "فجوة التعميم", "التقييم المركب", "حالة الأداء"
        ]
        
        leaderboard_data = [[cls._process_text(h, is_arabic, font_registered) for h in headers]]
        
        for row in leaderboard:
            # Safely fetch metric
            val_m = row.get("val_metrics", {})
            s_val = val_m.get(metric_key, 0.0)
            
            # Create row
            leaderboard_data.append([
                str(row.get("model_name", "Unknown")),
                f"{s_val:.4f}",
                f"{row.get('cv_mean', 0.0):.4f}",
                f"{row.get('cv_std', 0.0):.4f}",
                f"{row.get('generalization_gap', 0.0):.4f}",
                f"{row.get('composite_score', 0.0):.4f}",
                cls._process_text(str(row.get("status_indicator", "Stable")), is_arabic, font_registered)
            ])
            
        col_widths = [120, 60, 60, 60, 60, 60, 100]
        lb_table = Table(leaderboard_data, colWidths=col_widths)
        lb_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), bold_font_name),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DBEAFE")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#1E3A8A")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
            ('FONTNAME', (0,1), (-1,-1), font_name),
            ('FONTSIZE', (0,1), (-1,-1), 8),
        ]))
        story.append(lb_table)
        story.append(Spacer(1, 10))
        
        # Embed Leaderboard Chart if exists in dictionary
        temp_files = []
        if "leaderboard_comparison" in visualizations_dict:
            try:
                # Write to temp file
                tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tf.write(visualizations_dict["leaderboard_comparison"])
                tf.close()
                temp_files.append(tf.name)
                
                # Pre-validate image bytes using PIL
                with PILImage.open(tf.name) as img_check:
                    img_check.verify()
                
                story.append(KeepTogether([
                    Paragraph(cls._process_text("Model Leaderboard Visual Comparison" if not is_arabic else "المقارنة البصرية لمتصدري النماذج", is_arabic, font_registered), bold_body_style),
                    Image(tf.name, width=320, height=220),
                    Spacer(1, 10)
                ]))
            except Exception as chart_err:
                print(f"Failed to embed leaderboard comparison chart in PDF: {chart_err}")
                
        # Page break to look professional
        story.append(PageBreak())
        
        # Section 3: Champion Deep Analysis
        story.append(Paragraph(cls._process_text(t["best_model_title"], is_arabic, font_registered), h1_style))
        story.append(Paragraph(cls._process_text(t["best_model_desc"], is_arabic, font_registered), bold_body_style))
        
        # Detailed metrics panel
        best_row = leaderboard[0] if leaderboard else {}
        val_m = best_row.get("val_metrics", {})
        train_m = best_row.get("train_metrics", {})
        
        champion_data = [
            [cls._process_text("Performance Evaluation Metric" if not is_arabic else "مقياس الأداء والتقييم", is_arabic, font_registered),
             cls._process_text("Train Set" if not is_arabic else "بيانات التدريب", is_arabic, font_registered),
             cls._process_text("Validation Set" if not is_arabic else "بيانات التحقق", is_arabic, font_registered)]
        ]
        
        for k in val_m.keys():
            lbl = k.upper().replace("_", " ")
            v_train = train_m.get(k, 0.0)
            v_val = val_m.get(k, 0.0)
            champion_data.append([lbl, f"{v_train:.5f}", f"{v_val:.5f}"])
            
        cp_table = Table(champion_data, colWidths=[220, 150, 150])
        cp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#1F2937")),
            ('FONTNAME', (0,0), (-1,0), bold_font_name),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D1D5DB")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
            ('FONTNAME', (0,1), (-1,-1), font_name),
            ('FONTSIZE', (0,1), (-1,-1), 8),
        ]))
        story.append(cp_table)
        story.append(Spacer(1, 10))
        
        # Embed Confusion Matrix / Residuals plot
        chart_key = "confusion_matrix" if is_cls else "residual_plot"
        chart_title = ("Confusion Matrix Analysis" if is_cls else "Residual Distribution Analysis") if not is_arabic else ("تحليل مصفوفة الارتباك" if is_cls else "تحليل البواقي والأخطاء")
        
        if chart_key in visualizations_dict:
            try:
                tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tf.write(visualizations_dict[chart_key])
                tf.close()
                temp_files.append(tf.name)
                
                # Pre-validate image bytes using PIL
                with PILImage.open(tf.name) as img_check:
                    img_check.verify()
                
                story.append(KeepTogether([
                    Paragraph(cls._process_text(chart_title, is_arabic, font_registered), bold_body_style),
                    Image(tf.name, width=320, height=220),
                    Spacer(1, 10)
                ]))
            except Exception as chart_err:
                print(f"Failed to embed confusion matrix/residual chart in PDF: {chart_err}")
                
        # Section 4: Feature Importance
        if feature_importance:
            story.append(Paragraph(cls._process_text(t["importance_title"], is_arabic, font_registered), h1_style))
            
            imp_headers = ["Feature Column", "Relative Influence Impact"] if not is_arabic else ["اسم العمود الخصيصة", "تأثير وأهمية العمود"]
            imp_data = [[cls._process_text(h, is_arabic, font_registered) for h in imp_headers]]
            
            for item in feature_importance[:8]:
                f = item.get("feature", "")
                imp_val = item.get("importance", 0.0)
                imp_data.append([
                    cls._process_text(f, is_arabic, font_registered) if is_arabic else f,
                    f"{imp_val * 100:.2f}%"
                ])
                
            imp_table = Table(imp_data, colWidths=[260, 260])
            imp_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F9FAFB")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#374151")),
                ('FONTNAME', (0,0), (-1,0), bold_font_name),
                ('FONTSIZE', (0,0), (-1,0), 8),
                ('ALIGN', (0,0), (-1,-1), 'RIGHT' if is_arabic else 'LEFT'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F3F4F6")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB")),
                ('FONTNAME', (0,1), (-1,-1), font_name),
                ('FONTSIZE', (0,1), (-1,-1), 8),
            ]))
            story.append(imp_table)
            story.append(Spacer(1, 10))
            
        # Section 5: Recommendations
        story.append(Paragraph(cls._process_text(t["recommend_title"], is_arabic, font_registered), h1_style))
        story.append(Paragraph(cls._process_text(t["rec_1"], is_arabic, font_registered), body_style))
        story.append(Paragraph(cls._process_text(t["rec_2"], is_arabic, font_registered), body_style))
        story.append(Paragraph(cls._process_text(t["rec_3"], is_arabic, font_registered), body_style))
        story.append(Paragraph(cls._process_text(t["rec_4"], is_arabic, font_registered), body_style))
        
        # Build Document
        try:
            doc.build(story)
        finally:
            # Clean up temp files
            for tf_path in temp_files:
                try:
                    os.remove(tf_path)
                except Exception:
                    pass
                    
        pdf_buffer.seek(0)
        return pdf_buffer
