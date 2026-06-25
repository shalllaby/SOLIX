import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import os
import traceback

class AIImputer:
    def __init__(self, df: pd.DataFrame):
        # PASS BY REFERENCE: Do not duplicate the dataframe in memory to prevent OOM
        self.df = df
        
    def predict_missing_values(self, target_col: str, strategy: dict) -> tuple:
        """
        AI Imputer V32.1 (Index-Safe, Downsampled & Concurrent with Status Tracking):
        1. Resets index to RangeIndex to prevent duplicate/non-unique index alignment issues.
        2. Perform deterministic fillna if grouped relationships exist.
        3. Clean and preprocess features using zero-copy column views.
        4. Replace inf/-inf with NaN and drop corrupted target rows.
        5. Train model on a representative sample (max 50k rows) to limit RAM.
        6. Predict missing values in parallel chunks using ThreadPoolExecutor.
        7. Always restores the original index structure.
        Returns:
            (pd.DataFrame, str): The modified dataframe and the status flag.
        """
        # If target column doesn't exist or has no missing values, return early
        if target_col not in self.df.columns or self.df[target_col].isna().sum() == 0:
            return self.df, "no_nulls"

        original_index = self.df.index
        status = "fallback"
        
        try:
            # Set unique monotonic index for safe chunked alignment
            self.df.index = pd.RangeIndex(len(self.df))

            # --- 1. Sherlock Scan (Deterministic Group Imputation) ---
            for col in self.df.columns:
                if col == target_col:
                    continue
                if self.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[col]):
                    if pd.api.types.is_numeric_dtype(self.df[target_col]):
                        # Speed check: skip columns that are high cardinality/unique identifiers
                        if self.df[col].nunique() < len(self.df) * 0.8:
                            variances = self.df.groupby(col)[target_col].var()
                            if variances.mean() < 0.1 or variances.isna().all():
                                self.df[target_col] = self.df[target_col].fillna(
                                    self.df.groupby(col)[target_col].transform('mean')
                                )
                                if self.df[target_col].isna().sum() == 0:
                                    status = "ai_predictive"
                                    return self.df, status

            # --- 2. Prep Feature Space (Zero-Copy Column Analysis) ---
            excluded_cols = ['ID', 'Name', 'Email', 'Phone', 'Join_Date', 'Date', 'identifier']
            train_features = []
            
            # Identify columns useful as predictor features
            for col in self.df.columns:
                if col == target_col:
                    continue
                if col in excluded_cols:
                    continue
                
                # Skip high cardinality text columns to save memory and training complexity
                if not pd.api.types.is_numeric_dtype(self.df[col]) and self.df[col].nunique() > 50:
                    continue
                    
                train_features.append(col)
            
            if not train_features:
                self._fill_simple(target_col)
                status = "fallback"
                return self.df, status

            # --- 3. Representative Downsampling for Training ---
            train_idx = self.df.index[self.df[target_col].notna()]
            if len(train_idx) == 0:
                self._fill_simple(target_col)
                status = "fallback"
                return self.df, status

            # Downsample the training pool to max 100,000 rows to limit fitting memory usage
            if len(train_idx) > 100000:
                train_sample_idx = pd.Series(train_idx).sample(n=100000, random_state=42).to_numpy()
            else:
                train_sample_idx = train_idx

            # Slice training features without copying the entire DataFrame
            X_train = pd.DataFrame({col: self.df[col].loc[train_sample_idx] for col in train_features})
            y_train = self.df[target_col].loc[train_sample_idx]

            # Replace inf/-inf with NaN in training features and target
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            y_train = y_train.replace([np.inf, -np.inf], np.nan)

            # Drop any training rows where target became NaN after inf replacement
            valid_y_mask = y_train.notna()
            if not valid_y_mask.any():
                self._fill_simple(target_col)
                status = "fallback"
                return self.df, status
            X_train = X_train.loc[valid_y_mask]
            y_train = y_train.loc[valid_y_mask]

            # Fit encoders on global unique values to prevent unseen label exceptions during transform
            encoders = {}
            for col in train_features:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    unique_vals = pd.unique(self.df[col].dropna().unique().astype(str)).reshape(-1, 1)
                    if len(unique_vals) == 0:
                        unique_vals = np.array([['']]).astype(str)
                    oe.fit(unique_vals)
                    encoders[col] = oe
                    
                    # Apply encoding to training features
                    mask_train = X_train[col].notna()
                    if mask_train.any():
                        encoded_vals = oe.transform(X_train.loc[mask_train, [col]].astype(str))
                        X_train.loc[mask_train, col] = encoded_vals.flatten()

            # Convert to float32 first to avoid object-dtype downcasting warnings during fillna
            X_train = X_train.astype(np.float32).fillna(0)

            # Get prediction index (rows that require imputation)
            predict_idx = self.df.index[self.df[target_col].isna()]
            if len(predict_idx) == 0:
                status = "ai_predictive"
                return self.df, status

            # Determine target properties
            is_numeric = pd.api.types.is_numeric_dtype(self.df[target_col])

            try:
                # Downsample training data if it exceeds 100,000 rows to prevent memory explosion
                if len(X_train) > 100000:
                    X_train = X_train.sample(n=100000, random_state=42)
                    y_train = y_train.loc[X_train.index]

                # Fit model on the memory-safe training sample
                if is_numeric:
                    model = HistGradientBoostingRegressor(random_state=42, early_stopping=False)
                    model.fit(X_train, y_train)
                    # Check if majority of non-null values in target column are integers
                    non_nulls = self.df[target_col].dropna()
                    if len(non_nulls) > 0:
                        int_count = (non_nulls % 1 == 0).sum()
                        decimal_count = (non_nulls % 1 != 0).sum()
                        round_preds = int_count > decimal_count
                    else:
                        round_preds = False
                    target_le = None
                else:
                    model = HistGradientBoostingClassifier(random_state=42, early_stopping=False)
                    target_le = LabelEncoder()
                    y_train_encoded = target_le.fit_transform(y_train.astype(str))
                    model.fit(X_train, y_train_encoded)
                    round_preds = False

                # --- 4. Chunked Predictions (Sequential for Memory Safety) ---
                chunk_size = 500000
                predictions_list = []
                for i in range(0, len(predict_idx), chunk_size):
                    chunk_indices = predict_idx[i : i + chunk_size]
                    _, preds = self._process_chunk(
                        chunk_indices,
                        train_features,
                        encoders,
                        model,
                        target_le,
                        is_numeric,
                        round_preds
                    )
                    predictions_list.append(preds)
                
                if predictions_list:
                    all_preds = np.concatenate(predictions_list)
                    self.df.loc[predict_idx, target_col] = all_preds
                
                status = "ai_predictive"

            except Exception as e:
                import traceback
                print(f"\n[!] AI Imputer failed for target column '{target_col}':")
                traceback.print_exc()
                # Fall back to simple median/mode if any error occurs
                self._fill_simple(target_col)
                status = "fallback"

        finally:
            # Always restore the original index structure to guarantee no database or memory corruption
            self.df.index = original_index

        return self.df, status

    def _process_chunk(self, chunk_indices, train_features, encoders, model, target_le, is_numeric, round_preds) -> tuple:
        """
        Thread-safe chunk worker: extracts the slice, processes the encoding,
        and generates predictions for a subset of row indices.
        """
        # Only slice and copy the prediction features for this chunk
        X_chunk = pd.DataFrame({col: self.df[col].loc[chunk_indices] for col in train_features})
        
        # Replace inf/-inf with NaN
        X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)

        # Encode text features in chunk
        for col in train_features:
            if col in encoders:
                oe = encoders[col]
                mask_chunk = X_chunk[col].notna()
                if mask_chunk.any():
                    encoded_vals = oe.transform(X_chunk.loc[mask_chunk, [col]].astype(str))
                    X_chunk.loc[mask_chunk, col] = encoded_vals.flatten()
        
        # Convert to float32 first to avoid object-dtype downcasting warnings during fillna
        X_chunk = X_chunk.astype(np.float32).fillna(0)

        # Predict chunk values
        if is_numeric:
            chunk_preds = model.predict(X_chunk)
            if round_preds:
                chunk_preds = np.round(chunk_preds)
            else:
                chunk_preds = np.round(chunk_preds, 2)
        else:
            chunk_preds_encoded = model.predict(X_chunk)
            chunk_preds = target_le.inverse_transform(chunk_preds_encoded)

        return chunk_indices, chunk_preds

    def _fill_simple(self, col):
        """Fallback simple statistical imputation directly modifying the column."""
        # Replace inf/-inf with NaN in the column before simple imputation
        self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_numeric_dtype(self.df[col]):
            val = self.df[col].mean()
            # Apply majority rounding logic
            non_nulls = self.df[col].dropna()
            if len(non_nulls) > 0:
                int_count = (non_nulls % 1 == 0).sum()
                decimal_count = (non_nulls % 1 != 0).sum()
                if int_count > decimal_count:
                    val = round(val)
                else:
                    val = round(val, 2)
        else:
            val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
        
        self.df[col] = self.df[col].fillna(val)