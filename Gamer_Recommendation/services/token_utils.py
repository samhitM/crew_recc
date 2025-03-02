from core.repository import verify_token
from core.config import JWT_SECRET_KEY
from fastapi import HTTPException, Header
from typing import Optional
import jwt
import datetime


def validate_token(token: Optional[str] = Header(None)) -> str:
    """
    Validates the provided JWT token and extracts the userId.
    """
    if not token:
        raise HTTPException(status_code=403, detail="Forbidden: Missing token.")
    
    payload = verify_token(token)  # Decode token and get userId
    return payload["userId"]  # Extract and return userId

def generate_jwt_token(user_id):
        """
        Generates a JWT token for authentication.
        """
        payload = {
            "userId": user_id, 
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)  # 24-hour expiry
        }
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
        return token

