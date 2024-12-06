from core.repository import verify_token
import core.config as config
from fastapi import HTTPException
from typing import Optional


def validate_secret_key(secret_key: Optional[str]):
    """Verifies the provided secret key token and compares it with the configured secret key."""
    keys = verify_token(secret_key)
    if keys != config.SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret key")
