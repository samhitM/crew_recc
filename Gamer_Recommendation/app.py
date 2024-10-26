from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Union
import tensorflow as tf
import numpy as np
import json
import jwt as pyjwt 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
from fastapi import HTTPException, Depends
import asyncio
from fastapi.concurrency import run_in_threadpool


# Import classes from the converted recommendation.py
from recommendation import (
    RecommendationSystem, DataPreprocessor, DatasetPreparer, MappingLayer, DataLoader, SiameseRecommendationModel  
)

# Initialize FastAPI app

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql
import httpx

# Database connection settings
DB_HOST = '34.44.52.84'
DB_PORT = 5432
DATABASE = 'crewdb'
DB_USER = 'admin_crew'
DB_PASSWORD = 'xV/nI2+=uOI&KL1P'

# Initialize FastAPI app
app = FastAPI()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql

# Database connection settings
DB_HOST = '34.44.52.84'
DB_PORT = 5432
DATABASE = 'crewdb'
DB_USER = 'admin_crew'
DB_PASSWORD = 'xV/nI2+=uOI&KL1P'

# Initialize FastAPI app
app = FastAPI()

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

@app.get("/get-endpoint-data")
def get_endpoint_data():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Query to fetch user_id and endpoint_data for all users
            query = sql.SQL("""
                SELECT endpoint_data, user_id
                FROM source_endpoint
                WHERE endpoint = %s
            """)
            cur.execute(query, ('getCompletePlayerData',))
            results = cur.fetchall()

            if results:
                # Transform results to match the desired JSON structure
                players_stats = []
                for row in results:
                    endpoint_data, user_id = row
                    # Example JSON structure for endpoint_data
                    data = {
                        "userId": user_id,
                        "endpoint_data": {
                            "endpointData": endpoint_data  # endpoint_data assumed to be in desired format
                        }
                    }
                    players_stats.append(data)
                
                return {"playersStats": players_stats}  # Return structured JSON with all users' data

            else:
                raise HTTPException(status_code=404, detail="No data found for the specified endpoint")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    
    finally:
        conn.close()



# Load the trained Siamese model
model = tf.keras.models.load_model(
    'models/siamese_model.keras',
    custom_objects={'SiameseRecommendationModel': SiameseRecommendationModel}
)

# Load data and prepare components (initially empty)
preprocessor = DataPreprocessor()
preprocessed_data = {}
mapping_layer = None
dataset_preparer = DatasetPreparer()

# Initialize RecommendationSystem (initially without data)
recommendation_system = RecommendationSystem(
    model=model, vae_model=None, 
    dataset_preparer=dataset_preparer, 
    mapping_layer=None
)

# Request schema for FastAPI
class RecommendationRequest(BaseModel):
    game_id: int
    user_id: str
    offset: int
    num_recommendations: Optional[int] = 20
    filters: Optional[dict] = None

# Dependency to verify token
security = HTTPBearer()

def verify_token(token) -> int:
    secret_key = "942bb82561c615e0d67a27538c8e203ea60423c0508956a134e4f747d88189bd"
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

# API endpoint to load data

# Modify the API endpoint to fetch data.json
# @app.post("/api/user-game-dataset/")
# async def load_data(
#     authorization: str = Header(None)
# ):
#     try:
#         # Define the external endpoint to fetch data.json
#         url = "https://94da-122-171-109-17.ngrok-free.app/api/players/steam/summaryDB"
        
#         # Set up the headers, including the Authorization token
#         headers = {
#             "Authorization": authorization,  # Pass the Authorization header
#             "Content-Type": "application/json"
#         }
        
#         # Send the GET request to the external endpoint
#         response = requests.get(url, headers=headers)
        
#         # Check if the request was successful
#         if response.status_code != 200:
#             raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from external source.")
        
#         # Extract the JSON data
#         raw_data = response.json()
#         print(raw_data)
#         players_stats = raw_data.get('playersStats', {})
        
#         global preprocessed_data, mapping_layer, recommendation_system
#         preprocessed_data = preprocessor.preprocess(players_stats)
#         mapping_layer = MappingLayer(preprocessed_data)
        
#         # Reinitialize the recommendation system with updated data
#         recommendation_system = RecommendationSystem(
#             model=model, vae_model=None, 
#             dataset_preparer=dataset_preparer, 
#             mapping_layer=mapping_layer
#         )
        
#         return {"message": "Data fetched and loaded successfully!"}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user-game-dataset/")
async def load_data():
    try:
        # Define the external endpoint to fetch data.json
        response =  get_endpoint_data()
        
        # Send the GET request using async httpx client, allowing redirects
        #async with httpx.AsyncClient(follow_redirects=True) as client:
            #response = await client.get(url)
        
        # Check if the request was successful
        #if response.status_code != 200:
            #raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from external source.")
        
        # Extract the JSON data
        #raw_data = response.json()
        players_stats = response.get('playersStats', {})
        
        global preprocessed_data, mapping_layer, recommendation_system
        preprocessed_data = preprocessor.preprocess(players_stats)
        mapping_layer = MappingLayer(preprocessed_data)
        
        # Reinitialize the recommendation system with updated data
        recommendation_system = RecommendationSystem(
            model=model, vae_model=None, 
            dataset_preparer=dataset_preparer, 
            mapping_layer=mapping_layer
        )
        return {"message": "Data fetched and loaded successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# API endpoint to get recommendations

@app.post("/api/game-user-recommendations/")
async def get_recommendations(
    request: RecommendationRequest,
    secret_key: Optional[str] = Header(None),  # Accept secret key in header
    #crew_user_id: int = Depends(verify_token)  # This will provide the decoded user_id
):
    keys=verify_token(secret_key)
    
    # Check if the secret key matches
    if keys != '7irNPR6kOia':
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret key")

    #print(crew_user_id)
    # if crew_user_id != request.user_id:
    #     raise HTTPException(status_code=403, detail="Unauthorized request: User ID mismatch")
    
    # Automatically load data if not already loaded
    global preprocessed_data, mapping_layer, recommendation_system
    if not preprocessed_data:
        load_data_response = await load_data()
        if "message" not in load_data_response:
            raise HTTPException(status_code=500, detail="Failed to auto-load data.")

    try:
        # Proceed with recommendations
        top_users = recommendation_system.recommend_top_users(
            preprocessed_data,
            request.game_id,
            request.user_id,
            request.offset,
            num_recommendations=request.num_recommendations,
            filters=request.filters
        )
        return {"recommended_users": top_users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Recommendation System API is running!"}


# from kafka import KafkaProducer

# producer = KafkaProducer(bootstrap_servers='kafka.default.svc.cluster.local:9092')

# @app.post("/send-message/")
# async def send_message_to_kafka(message: str):
#     producer.send('fastapi-topic', message.encode('utf-8'))
#     return {"message": "Message sent to Kafka"}