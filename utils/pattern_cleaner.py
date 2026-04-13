import pandas as pd
import numpy as np
import re

class PatternCleaner:
    @staticmethod
    def clean_email_column(series: pd.Series) -> pd.Series:
        """
        تنظيف الإيميلات:
        - توحيد الحروف (Lowercase).
        - حذف المسافات.
        - التأكد من الصيغة (Regex).
        - تحويل الإيميلات غير الصالحة لـ NaN.
        """
        # تحويل لـ String وتنظيف مبدئي
        clean_series = series.astype(str).str.lower().str.strip()
        
        # Regex بسيط وفعال للإيميلات
        # (حروف أو أرقام أو نقط) @ (حروف) . (حروف)
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        
        def validate(val):
            if val == 'nan' or val == 'none' or pd.isna(val):
                return np.nan
            if re.match(email_regex, val):
                return val
            return np.nan # لو مش إيميل (زي ahmed#gmail)، امسحه عشان يتعوض صح

        return clean_series.apply(validate)

    @staticmethod
    def clean_phone_column(series: pd.Series) -> pd.Series:
        """
        تنظيف أرقام الهواتف:
        - حذف الرموز المزعجة (-, (), space, .).
        - الاحتفاظ بالأرقام فقط (و علامة + في البداية).
        """
        def clean(val):
            val_str = str(val).strip()
            if val_str == 'nan' or val_str == 'none' or pd.isna(val):
                return np.nan
            
            # حذف أي حاجة مش رقم أو +
            # مثال: (010) 123-456  ->  010123456
            clean_val = re.sub(r'[^\d+]', '', val_str)
            
            # لو الرقم قصير جداً (أقل من 4 أرقام) غالباً غلط أو كود
            if len(clean_val) < 4:
                return np.nan
                
            return clean_val

        return series.apply(clean)