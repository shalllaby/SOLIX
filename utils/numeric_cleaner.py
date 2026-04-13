import pandas as pd
import numpy as np
import re

class NumericCleaner:
    @staticmethod
    def clean_dirty_strings(series: pd.Series) -> pd.Series:
        """
        تنظيف الأرقام المختلطة بالنصوص (مثل '8,500 USD' -> 8500).
        """
        def clean_val(val):
            if pd.isna(val): return np.nan
            val_str = str(val)
            # الاحتفاظ بالأرقام، العلامة العشرية، والسالب
            clean = re.sub(r'[^\d.-]', '', val_str)
            try:
                return float(clean)
            except:
                return np.nan

        return series.apply(clean_val)

    @staticmethod
    def remove_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0):
        """
        حذف القيم المتطرفة باستخدام Z-Score.
        """
        if col not in df.columns: return df, 0
        
        col_data = df[col].dropna()
        if len(col_data) == 0: return df, 0
        
        mean = col_data.mean()
        std = col_data.std()
        
        if std == 0: return df, 0
        
        z_scores = np.abs((df[col] - mean) / std)
        
        # تحديد القيم المتطرفة
        outliers_mask = z_scores > threshold
        outliers_count = outliers_mask.sum()
        
        # تحويل المتطرف لـ NaN عشان الـ AI يعوضه
        df.loc[outliers_mask, col] = np.nan
        
        return df, outliers_count

    @staticmethod
    def enforce_logical_bounds(series: pd.Series, col_name: str) -> pd.Series:
        """
        جديد: التأكد من منطقية البيانات (مثلاً: العمر والمرتب لا يمكن أن يكونوا بالسالب).
        """
        col_lower = col_name.lower()
        
        # قائمة الكلمات الدالة على كميات موجبة
        positive_keywords = ['age', 'salary', 'price', 'cost', 'amount', 'year', 'experience', 'score']
        
        if any(keyword in col_lower for keyword in positive_keywords):
            # تحويل القيم السالبة إلى القيمة المطلقة (Absolute)
            # مثال: -5 سنين تبقى 5 سنين (غالباً خطأ كتابة)
            return series.abs()
            
            # أو ممكن نخليها صفر لو نفضل ذلك:
            # return series.clip(lower=0)
            
        return series