import jwt as pyjwt
from fastapi import HTTPException
import core.config as config

def verify_token(token) -> str:
    # Attempt to decode the provided JWT token
    try:
        payload = pyjwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        
        # Extract crew user ID from token payload
        crew_user_id = payload.get("userId")
        
        # Raise an error if crew user ID is not present in the token
        if crew_user_id is None:
            raise HTTPException(status_code=403, detail="Unauthorized: crew user_id missing.")
        
        # Return the extracted crew user ID if validation is successful
        return crew_user_id

    # Handle expired token error
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    
    # Handle any other invalid token errors
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")
