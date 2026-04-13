import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class AIImputer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def predict_missing_values(self, target_col: str, strategy: dict) -> pd.DataFrame:
        """
        AI Imputer V28.0 (Fixed Encoding Leakage):
        1. Sherlock Scan (زي ما هو).
        2. AI Prediction مع حماية الأعمدة النصية من التحول لأرقام في الملف النهائي.
        """
        # لو العمود مش موجود أو مفيهوش قيم مفقودة
        if target_col not in self.df.columns or self.df[target_col].isna().sum() == 0:
            return self.df

        # --- 1. Sherlock Scan (المنطق الحتمي) ---
        # (نفس الكود القديم هنا مفيهوش مشكلة)
        for col in self.df.columns:
            if col == target_col: continue
            if self.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[col]):
                if pd.api.types.is_numeric_dtype(self.df[target_col]):
                    # التأكد إن العمود مش كله قيم فريدة زي الإيميل (عشان مياخدش وقت)
                    if self.df[col].nunique() < len(self.df) * 0.8: 
                        variances = self.df.groupby(col)[target_col].var()
                        if variances.mean() < 0.1 or variances.isna().all():
                            self.df[target_col] = self.df[target_col].fillna(
                                self.df.groupby(col)[target_col].transform('mean')
                            )
                            if self.df[target_col].isna().sum() == 0:
                                return self.df

        # --- 2. التحضير للـ AI (التعديل الخطير هنا) 🛡️ ---
        
        # نحدد الأعمدة "الممنوعة" من التدريب (High Cardinality)
        # الإيميل، الاسم، التواريخ، والـ ID.. دول بيلخبطوا الموديل وبيتحولوا لأرقام غلط
        excluded_cols = ['ID', 'Name', 'Email', 'Phone', 'Join_Date', 'Date', 'identifier']
        
        # نختار بس الأعمدة المفيدة للتدريب (رقمية أو فئات قليلة زي القسم والوظيفة)
        train_features = []
        for col in self.df.columns:
            if col == target_col: continue
            if col in excluded_cols: continue
            
            # لو نص بس فيه قيم كتير أوي (زي الإيميل)، تجاهله
            if self.df[col].dtype == 'object' and self.df[col].nunique() > 50:
                continue
                
            train_features.append(col)
        
        if not train_features:
            self._fill_simple(target_col)
            return self.df

        # نشتغل على نسخة مؤقتة عشان منبوظش الداتا الأصلية
        df_temp = self.df[train_features + [target_col]].copy()
        
        # تشفير النصوص (Encoding) في النسخة المؤقتة بس
        encoders = {}
        for col in train_features:
            if df_temp[col].dtype == 'object':
                le = LabelEncoder()
                # تحويل النصوص لأرقام مع تجاهل الـ NaNs مؤقتاً
                mask = df_temp[col].notna()
                df_temp.loc[mask, col] = le.fit_transform(df_temp.loc[mask, col].astype(str))
                encoders[col] = le

        # تقسيم الداتا لـ Train و Predict
        train_df = df_temp[df_temp[target_col].notna()]
        predict_df = df_temp[df_temp[target_col].isna()]
        
        if train_df.empty or predict_df.empty:
            self._fill_simple(target_col)
            return self.df

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_predict = predict_df.drop(columns=[target_col])
        
        # التأكد إن كل الداتا أرقام (تنظيف أخير للنسخة المؤقتة)
        X_train = X_train.fillna(0)
        X_predict = X_predict.fillna(0)

        try:
            # لو العمود المستهدف رقمي -> Regressor
            if pd.api.types.is_numeric_dtype(self.df[target_col]):
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                predicted_values = model.predict(X_predict)
                
                # تقريب الأرقام لو هي أصلاً صحيحة
                if (y_train % 1 == 0).all():
                    predicted_values = np.round(predicted_values)
                
            # لو العمود المستهدف نصي (زي القسم) -> Classifier
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                # لازم نحول الهدف لأرقام برضو عشان الموديل يفهمه
                target_le = LabelEncoder()
                y_train_encoded = target_le.fit_transform(y_train.astype(str))
                
                model.fit(X_train, y_train_encoded)
                predicted_encoded = model.predict(X_predict)
                
                # نرجع الأرقام لنصوص تاني (Decode)
                predicted_values = target_le.inverse_transform(predicted_encoded)

            # اللحظة الحاسمة: وضع القيم المتوقعة في الملف الأصلي (بدون ما نغير باقي الأعمدة)
            self.df.loc[self.df[target_col].isna(), target_col] = predicted_values
            
        except Exception as e:
            # print(f"AI Imputation failed for {target_col}: {e}")
            self._fill_simple(target_col)

        return self.df

    def _fill_simple(self, col, step=1):
        if pd.api.types.is_numeric_dtype(self.df[col]):
            val = self.df[col].mean()
        else:
            val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
        
        self.df[col] = self.df[col].fillna(val)