import jwt as pyjwt
from fastapi import HTTPException
from core.config import INTERNAL_SECRET_KEY

def verify_token(token: str) -> dict:
    """
    Decodes and verifies a JWT token, returning the payload.
    """
    try:
        print(token)
        payload = pyjwt.decode(token, INTERNAL_SECRET_KEY, algorithms=["HS256"])
        # print(payload)
        # print("Decoded Payload:", payload)  # Debugging output
        # if not payload.get("userId"):
        #     raise HTTPException(status_code=403, detail="Unauthorized: Missing userId in token.")

        return payload  # Return full payload

    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")