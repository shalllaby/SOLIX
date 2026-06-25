import os
import base64
import hashlib
from cryptography.fernet import Fernet

def _get_fernet_key() -> bytes:
    # Derive a 32-byte URL-safe base64-encoded key from SECRET_KEY or ENCRYPTION_KEY
    key_str = os.getenv("ENCRYPTION_KEY") or os.getenv("SECRET_KEY", "sol_super_secret_jwt_key_2026")
    hashed = hashlib.sha256(key_str.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(hashed)

_fernet = Fernet(_get_fernet_key())

def encrypt_value(val: str) -> str:
    if not val:
        return ""
    return _fernet.encrypt(val.encode("utf-8")).decode("utf-8")

def decrypt_value(val: str) -> str:
    if not val:
        return ""
    try:
        return _fernet.decrypt(val.encode("utf-8")).decode("utf-8")
    except Exception:
        # Return raw value in case of decryption failure (e.g. legacy cleartext data)
        return val
