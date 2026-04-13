import pandas as pd
from thefuzz import process, fuzz

class TextNormalizer:
    @staticmethod
    def fuzzy_fix_column(series: pd.Series, threshold: int = 88) -> pd.Series:
        """
        نسخة محسنة تمنع دمج الوظائف مثل Senior Engineer مع Engineer.
        """
        # 1. فلترة القيم الفارغة
        clean_series = series.copy()
        valid_mask = clean_series.notna() & clean_series.apply(lambda x: isinstance(x, str))
        
        if not valid_mask.any():
            return series

        # 2. تحليل التكرار
        counts = clean_series[valid_mask].value_counts()
        unique_strings = counts.index.tolist()
        
        # لو القيم الفريدة كتير جداً (أسماء مثلاً)، نتخطى عشان السرعة والدقة
        if len(unique_strings) > 500: 
            return series

        # 3. الخريطة الذكية
        mapping = {}
        processed = set()
        
        for base_word in unique_strings:
            if base_word in processed:
                continue
            
            # استخدام fuzz.ratio (دقيق) بدل الافتراضي
            matches = process.extract(base_word, unique_strings, limit=None, scorer=fuzz.ratio)
            
            for match_word, score in matches:
                if match_word == base_word:
                    continue
                
                # رفعنا النسبة لـ 88% لزيادة الأمان
                if score >= threshold:
                    # شرط: الكلمة الأصح هي الأكثر تكراراً
                    if counts[base_word] >= counts[match_word]:
                        if match_word not in mapping:
                            mapping[match_word] = base_word
                            processed.add(match_word)
            
            processed.add(base_word)

        # 4. التطبيق مع طباعة التغييرات للمراجعة
        if mapping:
            # print(f"   [Fuzzy Fix] Correcting {len(mapping)} typos: {list(mapping.items())[:3]}...")
            return clean_series.replace(mapping)
            
        return clean_series