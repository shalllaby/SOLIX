import pandas as pd
import concurrent.futures

try:
    from thefuzz import process, fuzz
except ModuleNotFoundError:
    import difflib
    
    class FakeFuzz:
        @staticmethod
        def ratio(s1: str, s2: str) -> int:
            if not s1 or not s2:
                return 0
            return int(difflib.SequenceMatcher(None, str(s1), str(s2)).ratio() * 100)
            
    class FakeProcess:
        @staticmethod
        def extract(query: str, choices: list, limit: int = None, scorer = None) -> list:
            if scorer is None:
                scorer = FakeFuzz.ratio
            results = []
            for choice in choices:
                score = scorer(query, choice)
                results.append((choice, score))
            results.sort(key=lambda x: x[1], reverse=True)
            if limit is not None:
                return results[:limit]
            return results
            
    fuzz = FakeFuzz
    process = FakeProcess


def _fuzzy_match_worker(args):
    """
    Top-level helper function for ProcessPoolExecutor to compute fuzzy matching scores.
    Must be defined at the module level to be serializable (picklable) across processes.
    """
    base_word, unique_strings, threshold = args
    matches = process.extract(base_word, unique_strings, limit=None, scorer=fuzz.ratio)
    valid_matches = []
    for match_word, score in matches:
        if match_word == base_word:
            continue
        if score >= threshold:
            valid_matches.append((match_word, score))
    return base_word, valid_matches


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
        
        # Optimize: Use ProcessPoolExecutor for larger string sets to exploit multi-core CPUs
        if len(unique_strings) >= 10:
            tasks = [(base_word, unique_strings, threshold) for base_word in unique_strings]
            matches_dict = {}
            
            # Using 4 workers to match Kaggle's 4-core environment
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                for base_word, valid_matches in executor.map(_fuzzy_match_worker, tasks):
                    matches_dict[base_word] = valid_matches
                    
            for base_word in unique_strings:
                if base_word in processed:
                    continue
                matches = matches_dict.get(base_word, [])
                for match_word, score in matches:
                    if counts[base_word] >= counts[match_word]:
                        if match_word not in mapping:
                            mapping[match_word] = base_word
                            processed.add(match_word)
                processed.add(base_word)
        else:
            # Sequential fallback for very small datasets to avoid multiprocessing overhead
            for base_word in unique_strings:
                if base_word in processed:
                    continue
                matches = process.extract(base_word, unique_strings, limit=None, scorer=fuzz.ratio)
                for match_word, score in matches:
                    if match_word == base_word:
                        continue
                    if score >= threshold:
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