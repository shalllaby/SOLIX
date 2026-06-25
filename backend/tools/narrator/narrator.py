# -*- coding: utf-8 -*-
import json
import os
import urllib.request
from typing import Optional


class DataNarrator:
    """
    SOL Data Narrator - generates Arabic reports explaining data before and after cleaning.
    Uses Groq Llama-3.3-70b for intelligent generation.
    """

    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or key == "your_api_key_here":
            key = ""
        self.api_key = key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    # ──────────────────────────────────────────────────────────────────
    def _call_llm(self, system_prompt: str, user_message: str, max_tokens: int = 1200) -> Optional[str]:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("Groq API key is not configured.")
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.5,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                try:
                    from backend.utils.llm_logger import log_groq_response
                    log_groq_response(body, module_name="narrator")
                except Exception as e_log:
                    print(f"Failed to log narrator token usage: {e_log}")
                return body["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Narrator LLM Error] {e}")
            return None

    # ──────────────────────────────────────────────────────────────────
    #  PART 1: Pre-Cleaning Narrative
    # ──────────────────────────────────────────────────────────────────
    def narrate_pre_cleaning(self, metadata: dict, filename: str = "dataset") -> str:
        cols       = metadata.get("columns_info", [])
        dirty_cols = [c for c in cols if (c.get("missing_count") or 0) > 0]
        row_count  = metadata.get("rows", "?")
        col_count  = metadata.get("cols", len(cols))

        col_details = "\n".join([
            f"- {c['name']} ({c.get('physical_type', '?')}): "
            f"{c.get('missing_count', 0)} missing/dirty values ({c.get('missing_percentage', 0):.1f}%)"
            for c in cols
        ])

        system_prompt = (
            'You are "SOL Data Narrator" - a data expert who explains technical information '
            "in clear, professional Arabic.\n\n"
            "Task: Write a narrative report in Arabic explaining the user's dataset before cleaning.\n\n"
            "Rules:\n"
            "- Write in simple formal Arabic (no dialect)\n"
            "- Do NOT use technical terms without explanation\n"
            "- Be specific: mention real column names and real numbers\n"
            "- Tone: encouraging and professional\n"
            "- Use Markdown with Arabic headings\n"
            "- Start directly with content, no lengthy introduction"
        )

        user_message = (
            f"Dataset file: {filename}\n"
            f"Contains {row_count} rows and {col_count} columns.\n\n"
            f"Column details:\n{col_details}\n\n"
            "Write the Arabic narrative report with these sections (use Markdown):\n"
            "1. ## نظرة عامة على بياناتك - brief engaging paragraph about the dataset and its likely purpose\n"
            "2. ## الأعمدة الأساسية وأهميتها - bullet points for each important column and its role\n"
            "3. ## العلاقات الخفية - describe potential correlations between columns\n"
            "4. ## تشخيص المشكلات - list affected columns with real numbers"
        )

        result = self._call_llm(system_prompt, user_message, max_tokens=1200)
        if result:
            return result

        # Fallback
        dirty_summary = ", ".join(
            [f"**{c['name']}** ({c.get('missing_count', 0)} values)" for c in dirty_cols[:4]]
        )
        return (
            f"## نظرة عامة على بياناتك\n\n"
            f"ملف **{filename}** يحتوي على **{row_count} سجل** موزعة على **{col_count} عمود**.\n\n"
            f"## تشخيص المشكلات\n\n"
            f"وُجدت قيم ناقصة أو تالفة في: {dirty_summary or 'لا توجد مشاكل ظاهرة'}.\n\n"
            f"> **إجمالي الخلايا المتضررة:** "
            f"{sum(c.get('missing_count', 0) for c in dirty_cols)} خلية\n"
        )

    # ──────────────────────────────────────────────────────────────────
    #  PART 2: Post-Cleaning Narrative
    # ──────────────────────────────────────────────────────────────────

    # Mapping of action keys to human-readable Arabic descriptions
    _METHOD_AR = {
        "smart_impute":     "تعبئة ذكية (Smart Impute) - تحليل الانماط المحيطة واختيار القيمة الانسب",
        "remove_outliers":  "ازالة القيم الشاذة (Z-Score) ثم تعبئة الفراغات بالوسيط الاحصائي",
        "fuzzy_fix":        "تصحيح اخطاء الكتابة (Fuzzy Matching) - مطابقة النصوص بأقرب قيمة صحيحة",
        "standardize_date": "توحيد صيغة التاريخ - تحويل كل الصيغ المختلفة لتنسيق موحد",
        "impute_mean":      "تعبئة بالمتوسط الحسابي (Mean Imputation)",
        "impute_mode":      "تعبئة بالقيمة الاكثر تكراراً (Mode Imputation)",
        "drop":             "حذف العمود لاحتوائه على نسبة تلف عالية",
        "clean_pattern":    "تنظيف صيغة البيانات (Pattern Cleaning) مثل البريد الالكتروني والهاتف",
    }

    _STRATEGY_AR = {
        "alpha": (
            "**الفا (Alpha)** - الاستراتيجية الحذرة: "
            "الحد الادنى من التعديلات والحفاظ على البيانات الاصلية قدر الامكان"
        ),
        "beta": (
            "**بيتا (Beta)** - الاستراتيجية المتوازنة: "
            "تنظيف شامل يشمل تعبئة القيم الناقصة واذالة المكررات ومعالجة القيم الشاذة"
        ),
        "gamma": (
            "**غاما (Gamma)** - الاستراتيجية الشاملة: "
            "تنظيف عدواني لضمان اعلى جودة يشمل حذف الصفوف المشبوهة وفرض الانواع الصحيحة"
        ),
    }

    def narrate_post_cleaning(
        self,
        strategy: str,
        stats: dict,
        report: dict,
        filename: str = "dataset",
    ) -> str:
        """Generate Arabic post-cleaning narrative with per-column action explanation."""

        rows_before    = stats.get("rows_before", "?")
        rows_after     = stats.get("rows_after",  "?")
        missing_before = stats.get("missing_before", "?")
        missing_after  = stats.get("missing_after",  "?")
        cells_fixed    = stats.get("cells_fixed", "?")

        strategy_ar   = self._STRATEGY_AR.get(strategy.lower(), f"**{strategy}**")
        cleaning_plan = report.get("cleaning_strategy", {})
        actions_log   = report.get("actions", [])

        # Build per-column table for the prompt
        col_table_lines = [
            f"| {col} | {self._METHOD_AR.get(action, action)} |"
            for col, action in cleaning_plan.items()
            if not col.startswith("__")
        ]
        col_table = "\n".join(col_table_lines) if col_table_lines else "No detailed plan available."

        actions_text = "\n".join([f"- {a}" for a in actions_log]) if actions_log else "No action log."

        system_prompt = (
            'You are "SOL Data Narrator". Write a professional post-cleaning report in Arabic.\n\n'
            "Rules:\n"
            "- Write in simple formal Arabic only\n"
            "- Mention each affected column by name with what happened to it\n"
            "- State the exact method used for each column clearly\n"
            "- Do NOT write a lengthy introduction\n"
            "- Use Markdown with Arabic headings, bullet points, and blockquotes"
        )

        user_message = (
            f"Dataset: **{filename}**\n"
            f"Strategy used: {strategy_ar}\n\n"
            f"Cleaning stats:\n"
            f"- Rows before: {rows_before} -> after: {rows_after}\n"
            f"- Missing/dirty values before: {missing_before} -> after: {missing_after}\n"
            f"- Total cells fixed: **{cells_fixed}**\n\n"
            f"Per-column cleaning plan (column | method used):\n"
            f"{col_table}\n\n"
            f"Execution log:\n{actions_text}\n\n"
            "Write the report in Arabic with exactly these sections:\n\n"
            "## ما الذي تغيّر في بياناتك؟\n"
            "[One paragraph summarizing the overall cleaning]\n\n"
            "## شرح تفصيلي لكل عمود\n"
            "[For each affected column: name, original problem, exact method used]\n\n"
            "## لماذا اخترنا هذه الاستراتيجية؟\n"
            "[2 sentences explaining the strategy choice]\n\n"
            "## الحكم النهائي\n"
            "[One definitive sentence: is the data ready? confidence level?]"
        )

        result = self._call_llm(system_prompt, user_message, max_tokens=1400)
        if result:
            return result

        # Fallback without LLM
        col_lines = "\n".join([
            f"- **{col}**: {self._METHOD_AR.get(action, action)}"
            for col, action in list(cleaning_plan.items())[:8]
            if not col.startswith("__")
        ]) or "- لا توجد تفاصيل متاحة."

        return (
            f"## ما الذي تغيّر في بياناتك؟\n\n"
            f"قام محرك SOL بمعالجة ملف **{filename}** باستخدام {strategy_ar}. "
            f"تم اصلاح **{cells_fixed} خلية** من اجمالي **{missing_before}** قيمة ناقصة او تالفة.\n\n"
            f"## شرح تفصيلي لكل عمود\n\n"
            f"{col_lines}\n\n"
            f"## الحكم النهائي\n\n"
            f"> البيانات جاهزة للانتاج - {rows_after} سجل نظيف وموثوق.\n"
        )
