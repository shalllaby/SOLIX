import pandas as pd
import os
from datetime import datetime
import json

class SystemLogger:
    def __init__(self, log_file="system_activity_log.xlsx"):
        self.log_file = log_file

    def log_operation(self, filename: str, file_rows: int, strategy: dict, processing_time: float, 
                      missing_before: int = 0, missing_after: int = 0, status: str = "Success"):
        """
        تسجيل العملية مع مقاييس الجودة (Quality Score)
        """
        # حساب نسبة التحسن (Quality Score)
        cleaned_count = missing_before - missing_after
        improvement_pct = 0.0
        if missing_before > 0:
            improvement_pct = round((cleaned_count / missing_before) * 100, 2)
        elif status == "Success":
            improvement_pct = 100.0 # الملف كان نظيفاً أصلاً

        # تجهيز البيانات
        new_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": filename,
            "Rows": file_rows,
            "Status": status,
            "Time (s)": round(processing_time, 2),
            "Missing_Before": missing_before,
            "Missing_After": missing_after,
            "Fixed_Cells": cleaned_count,
            "Quality_Score (%)": f"{improvement_pct}%",
            "Strategy_Summary": json.dumps(strategy.get("cleaning_strategy", {}), ensure_ascii=False)
        }

        # إنشاء DataFrame
        new_df = pd.DataFrame([new_entry])

        # الحفظ في الإكسيل
        if os.path.exists(self.log_file):
            try:
                existing_df = pd.read_excel(self.log_file)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_excel(self.log_file, index=False)
            except Exception as e:
                print(f"⚠️ Logger Error (Append): {e}")
        else:
            new_df.to_excel(self.log_file, index=False)
        
        print(f"✅ [Logger] Quality Report saved: Fixed {cleaned_count} cells ({improvement_pct}%)")