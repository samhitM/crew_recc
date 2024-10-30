from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Union
import tensorflow as tf
import psycopg2
from psycopg2 import sql
import jwt as pyjwt
from fastapi.security import HTTPBearer
import httpx

from recommendation import (
    RecommendationSystem, DataPreprocessor, DatasetPreparer, MappingLayer, SiameseRecommendationModel  
)

app = FastAPI()

# Database connection settings
DB_HOST = '34.44.52.84'
DB_PORT = 5432
DATABASE = 'crewdb'
DB_USER = 'admin_crew'
DB_PASSWORD = 'xV/nI2+=uOI&KL1P'

# Database connection function
def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DATABASE,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return connection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to the database: {str(e)}")

# Token verification
security = HTTPBearer()

def verify_token(token) -> str:
    secret_key = "1c75f472b1e52c582c8ed3f4d88af9c0137f9a2eeeb1d63e97ecedd1be8f1a3c"
    try:
        payload = pyjwt.decode(token, secret_key, algorithms=["HS256"])
        crew_user_id = payload.get("userId")
        if crew_user_id is None:
            raise HTTPException(status_code=403, detail="Unauthorized: crew user_id missing.")
        return crew_user_id
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

# Function to fetch data from the database
def get_endpoint_data():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = sql.SQL("""
                SELECT endpoint_data, user_id
                FROM source_endpoint
                WHERE endpoint = %s
            """)
            cur.execute(query, ('getCompletePlayerData',))
            results = cur.fetchall()
            if results:
                players_stats = [{"userId": row[1], "endpoint_data": {"endpointData": row[0]}} for row in results]
                return {"playersStats": players_stats}
            else:
                raise HTTPException(status_code=404, detail="No data found for the specified endpoint")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    finally:
        conn.close()

# Fetch username for a given user ID
def get_username(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT full_name FROM \"user\" WHERE id = %s;", (user_id,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching username: {str(e)}")
    finally:
        conn.close()

# Initialize components for recommendation system
model = tf.keras.models.load_model(
    'models/siamese_model.keras',
    custom_objects={'SiameseRecommendationModel': SiameseRecommendationModel}
)
preprocessor = DataPreprocessor()
dataset_preparer = DatasetPreparer()

class RecommendationCache:
    def __init__(self):
        self.preprocessed_data = None
        self.mapping_layer = None
        self.recommendation_system = None

recommendation_cache = RecommendationCache()

# Data loading endpoint
@app.get("/api/user-game-dataset/")
async def load_data():
    try:
        players_stats = get_endpoint_data().get('playersStats', {})
        recommendation_cache.preprocessed_data = preprocessor.preprocess(players_stats)
        recommendation_cache.mapping_layer = MappingLayer(recommendation_cache.preprocessed_data)
        recommendation_cache.recommendation_system = RecommendationSystem(
            model=model, vae_model=None,
            dataset_preparer=dataset_preparer,
            mapping_layer=recommendation_cache.mapping_layer
        )
        return {"message": "Data fetched and loaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

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
    secret_key: Optional[str] = Header(None)
):
    
    keys = verify_token(secret_key)
    
    # Check if the secret key matches
    if keys != '7irNPR6kOia':
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret key")

    if recommendation_cache.preprocessed_data is None:
        await load_data()

    try:
        top_users = recommendation_cache.recommendation_system.recommend_top_users(
            recommendation_cache.preprocessed_data,
            request.game_id,
            request.user_id,
            request.offset,
            num_recommendations=request.num_recommendations,
            filters=request.filters
        )

        # Fetch username for the requesting user_id
        requesting_username = get_username(request.user_id)
        
        # Include usernames for each recommended user
        for user in top_users["recommended_users"]:
            user["username"] = get_username(user["user_id"])

        # Prepare and return the response, including requesting user's username
        return {
            "game_id": request.game_id,
            "user_id": request.user_id,
            "username": requesting_username,
            "offset": request.offset,
            "recommended_users": top_users["recommended_users"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Recommendation System API is running!"}
