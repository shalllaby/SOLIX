import pandas as pd
import numpy as np
import warnings

class DateParser:
    @staticmethod
    def standardize_column(series: pd.Series) -> pd.Series:
        # V20 Update: تجاهل التحذيرات المزعجة
        warnings.filterwarnings("ignore", category=UserWarning) 
        
        try:
            # 1. تنظيف أولي
            temp_col = series.astype(str).str.strip().str.replace(r'[./\s\\]', '-', regex=True)
            
            final_dates = pd.Series(index=series.index, dtype='object')
            
            # 2. ISO
            iso_dates = pd.to_datetime(temp_col, format='%Y-%m-%d', errors='coerce')
            final_dates = final_dates.fillna(iso_dates)
            
            # 3. Egyptian (DMY)
            mask_missing = final_dates.isna()
            if mask_missing.any():
                dmy_dates = pd.to_datetime(temp_col[mask_missing], format='%d-%m-%Y', errors='coerce')
                final_dates = final_dates.fillna(dmy_dates)
                
                # Fallback
                mask_missing = final_dates.isna()
                if mask_missing.any():
                    dmy_general = pd.to_datetime(temp_col[mask_missing], dayfirst=True, errors='coerce')
                    final_dates = final_dates.fillna(dmy_general)
            
            # 4. Text
            mask_missing = final_dates.isna()
            if mask_missing.any():
                try:
                    text_dates = pd.to_datetime(temp_col[mask_missing], errors='coerce')
                    final_dates = final_dates.fillna(text_dates)
                except: pass

            return pd.to_datetime(final_dates, errors='coerce')
            
        except Exception as e:
            print(f"Date Parsing Error: {e}")
            return series