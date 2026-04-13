import pandas as pd

# Shared in-memory store for datasets across tools
_store: dict[str, pd.DataFrame] = {}
_store_ext: dict[str, str] = {}
_store_filename: dict[str, str] = {}
_store_goals: dict[str, str] = {}
_store_is_db: dict[str, bool] = {}

# Audit subsystem — maps dataset_id → structured AuditLog dict
_audit_store: dict[str, dict] = {}
