from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from core.config import CREW_USER_ID
from services.token_utils import validate_token
from services.recommendation_utils import load_data, compute_recommendations, attach_usernames, get_user_id_to_username_mapping, initialize_recommendation_system, periodic_model_training
from database.connection import init_db_pools, db_pools
from cache.recommendation_cache import recommendation_cache
from services.token_utils import generate_jwt_token
from crew_scoring.impressionScoring.cron.periodic_updater import periodic_crew_impression_update

import threading

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """Initialize database connection pools on app startup."""
    init_db_pools()
    """ Initialize recommendation system at startup """
    initialize_recommendation_system()
    threading.Thread(target=periodic_crew_impression_update, daemon=True).start()  # Start background crew impression updater
    threading.Thread(target=periodic_model_training, daemon=True).start() # Start periodic model retraining

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
    extracted_user_id = validate_token(token)  # Validate token and extract userId
    # Ensure the user ID in the request matches the one in the token
    if CREW_USER_ID != extracted_user_id:
        raise HTTPException(status_code=403, detail="Forbidden: user_id mismatch.")

    if recommendation_cache.preprocessed_data is None:
        print("Preprocessed data is missing. Loading now...")
        load_data()

    try:
        top_users = compute_recommendations(request)
        
        # Collect all user IDs (requesting user + recommended users)
        user_ids = {request.user_id} | {user["user_id"] for user in top_users["recommended_users"]}
        user_map = get_user_id_to_username_mapping(user_ids)
        requesting_username = user_map.get(request.user_id, None)

        # Attach usernames to recommended users
        attach_usernames(top_users, user_map)

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
    

@app.on_event("shutdown")
def shutdown_event():
    """Close all database pools when the app shuts down."""
    global db_pools
    for pool in db_pools.values():
        pool.closeall()

# Health check endpoint
@app.get("/")
def read_root():
    """Simple health check endpoint to verify API status."""
    return {"message": "Recommendation System API is running!"}
