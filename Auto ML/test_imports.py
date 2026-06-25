import os
import sys
import traceback

os.environ['POLARS_SKIP_CPU_CHECK'] = '1'

print("Python version:", sys.version)

try:
    import xgboost
    print("XGBoost version:", xgboost.__version__)
except Exception as e:
    print("XGBoost import failed:")
    traceback.print_exc()

try:
    import lightgbm
    print("LightGBM version:", lightgbm.__version__)
except Exception as e:
    print("LightGBM import failed:")
    traceback.print_exc()

try:
    import catboost
    print("CatBoost version:", catboost.__version__)
except Exception as e:
    print("CatBoost import failed:")
    traceback.print_exc()
