from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

from services.token_utils import validate_token
from services.recommendation_utils import load_data, compute_recommendations, attach_usernames
from database import get_username
from cache.recommendation_cache import recommendation_cache
from services.token_utils import generate_jwt_token

app = FastAPI()

# Recommendation request schema
class RecommendationRequest(BaseModel):
    game_id: int
    user_id: str
    offset: int
    num_recommendations: Optional[int] = 20
    filters: Optional[dict] = None

# Recommendation endpoint
@app.post("/api/game-user-recommendations/")
async def get_recommendations(
    request: RecommendationRequest,
    token: Optional[str] = Header(None)
):
    """Handles recommendation requests, validating the secret key, loading data if needed, and returning recommendations."""
    extracted_user_id = validate_token(generate_jwt_token(request.user_id))  # Validate token and extract userId
    
    # Ensure the user ID in the request matches the one in the token
    if request.user_id != extracted_user_id:
        raise HTTPException(status_code=403, detail="Forbidden: user_id mismatch.")

    if recommendation_cache.preprocessed_data is None:
        load_data()

    try:
        top_users = compute_recommendations(request)
        requesting_username = get_username(request.user_id)
        attach_usernames(top_users)

        return {
            "game_id": request.game_id,
            "user_id": request.user_id,
            "username": requesting_username,
            "offset": request.offset,
            "user_specialisations": top_users["user_specialization"],
            "recommended_users": top_users["recommended_users"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

# Health check endpoint
@app.get("/")
def read_root():
    """Simple health check endpoint to verify API status."""
    return {"message": "Recommendation System API is running!"}
