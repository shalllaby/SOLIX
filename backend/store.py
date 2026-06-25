import pandas as pd
from contextvars import ContextVar

# Context variable to hold the active user's ID during a request
active_user_id: ContextVar[int] = ContextVar("active_user_id", default=None)

class UserIsolatedDict(dict):
    def _get_user_prefix(self) -> str:
        uid = active_user_id.get()
        return f"user_{uid}:" if uid is not None else ""

    def _find_prefix_for_key(self, key: str) -> str:
        if not isinstance(key, str):
            return ""
            
        # 1. Check if the key itself already exists with a prefix in this dictionary
        for k in super().keys():
            if isinstance(k, str) and k.startswith("user_") and ":" in k:
                prefix, base_key = k.split(":", 1)
                if base_key == key:
                    return f"{prefix}:"
                    
        # 2. Check if the key contains any known base key from this dictionary
        all_existing = []
        for k in super().keys():
            if isinstance(k, str) and k.startswith("user_") and ":" in k:
                prefix, base_key = k.split(":", 1)
                all_existing.append((prefix, base_key))
        all_existing.sort(key=lambda x: len(x[1]), reverse=True)
        
        for prefix, base_key in all_existing:
            if base_key in key:
                return f"{prefix}:"
                
        # 3. Fallback: check other dictionaries in store module
        import backend.store as bs
        for other_dict_name in ("_store", "_store_parquet_path", "_store_tasks", "_store_filename"):
            other_dict = getattr(bs, other_dict_name, None)
            if other_dict is self or not isinstance(other_dict, UserIsolatedDict):
                continue
            for k in super(UserIsolatedDict, other_dict).keys():
                if isinstance(k, str) and k.startswith("user_") and ":" in k:
                    prefix, base_key = k.split(":", 1)
                    if base_key == key or base_key in key:
                        return f"{prefix}:"
                        
        return ""

    def _transform_key(self, key: str) -> str:
        if not isinstance(key, str):
            return key
        
        # If the key is already prefixed, return as-is
        if key.startswith("user_") and ":" in key:
            return key
            
        prefix = self._get_user_prefix()
        if prefix:
            # Strictly use the active user's prefix when inside a request context
            return f"{prefix}{key}"
            
        # If active_user_id is None (background task context), search for an existing prefix
        found_prefix = self._find_prefix_for_key(key)
        if found_prefix:
            return f"{found_prefix}{key}"
            
        return key

    def __getitem__(self, key):
        t_key = self._transform_key(key)
        return super().__getitem__(t_key)

    def __setitem__(self, key, value):
        t_key = self._transform_key(key)
        super().__setitem__(t_key, value)

    def __delitem__(self, key):
        t_key = self._transform_key(key)
        super().__delitem__(t_key)

    def __contains__(self, key):
        t_key = self._transform_key(key)
        return super().__contains__(t_key)

    def get(self, key, default=None):
        t_key = self._transform_key(key)
        return super().get(t_key, default)

    def pop(self, key, *args):
        t_key = self._transform_key(key)
        return super().pop(t_key, *args)

    def setdefault(self, key, default=None):
        t_key = self._transform_key(key)
        return super().setdefault(t_key, default)

    def clear(self):
        # Clear only the keys belonging to the active user
        prefix = self._get_user_prefix()
        if not prefix:
            super().clear()
            return
        keys_to_del = [k for k in super().keys() if isinstance(k, str) and k.startswith(prefix)]
        for k in keys_to_del:
            super().__delitem__(k)

    def keys(self):
        prefix = self._get_user_prefix()
        all_keys = super().keys()
        if not prefix:
            result = []
            for k in all_keys:
                if isinstance(k, str) and k.startswith("user_") and ":" in k:
                    result.append(k.split(":", 1)[1])
                else:
                    result.append(k)
            return result
        return [k[len(prefix):] for k in all_keys if isinstance(k, str) and k.startswith(prefix)]

    def items(self):
        prefix = self._get_user_prefix()
        all_items = super().items()
        if not prefix:
            result = []
            for k, v in all_items:
                if isinstance(k, str) and k.startswith("user_") and ":" in k:
                    result.append((k.split(":", 1)[1], v))
                else:
                    result.append((k, v))
            return result
        return [(k[len(prefix):], v) for k, v in all_items if isinstance(k, str) and k.startswith(prefix)]

    def values(self):
        prefix = self._get_user_prefix()
        if not prefix:
            return super().values()
        return [v for k, v in super().items() if isinstance(k, str) and k.startswith(prefix)]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in (other.items() if hasattr(other, 'items') else other):
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

# Shared in-memory store for datasets across tools
_store = UserIsolatedDict()
_store_parquet_path = UserIsolatedDict()
_store_tasks = UserIsolatedDict()
_store_ext = UserIsolatedDict()
_store_filename = UserIsolatedDict()
_store_goals = UserIsolatedDict()
_store_is_db = UserIsolatedDict()

# Audit subsystem — maps dataset_id → structured AuditLog dict
_audit_store = UserIsolatedDict()

# ─────────────────────────────────────────────────────────────
#  VizEngine stores
# ─────────────────────────────────────────────────────────────
# MODE_DISCOVERY  → built right after /api/upload  (raw data only)
_discovery_store = UserIsolatedDict()

# MODE_COMPARISON → built right after /api/clean   (raw vs cleaned)
_viz_store = UserIsolatedDict()
