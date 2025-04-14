from database import get_endpoint_data, fetch_all_users_data
from recommendation_system import RecommendationSystem
from data_preprocessing import DataPreprocessor, DatasetPreparer, MappingLayer
from models import SiameseRecommendationModel
from cache.recommendation_cache import recommendation_cache
from utils import ModelTrainer
import threading
import time
from fastapi import HTTPException
from datetime import datetime, timedelta
import pytz  # To handle timezones

# Initialize components for recommendation system
preprocessor = DataPreprocessor()
dataset_preparer = DatasetPreparer()

# Flag to track if model is trained
model_trained = False

# Function to load data and initialize recommendation system components
import time

def load_data():
    try:
        players_stats = get_endpoint_data().get('playersStats', {})
        recommendation_cache.update_preprocessed_data(preprocessor.preprocess(players_stats))
        recommendation_cache.update_mapping_layer(MappingLayer(recommendation_cache.preprocessed_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    

def initialize_recommendation_system():
    global model_trained
    try:
        print("Initializing recommendation system with pre-trained model...")
        load_data()
        
        # Prepare training and validation datasets
        train_dataset = dataset_preparer.prepare_tf_dataset(recommendation_cache.preprocessed_data,
                                                            recommendation_cache.mapping_layer)
        
        # Initialize and train a new model
        num_users = len(recommendation_cache.preprocessed_data['player_id'].unique())
        num_games = len(recommendation_cache.preprocessed_data['game_id'].unique())

        embedding_dim = 512
        siamese_model = SiameseRecommendationModel(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
        model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
        model_trainer.train_model(train_dataset, train_dataset)
        
        # Update the recommendation system with the new model
        recommendation_cache.update_recommendation_system(
            RecommendationSystem(
                model=siamese_model,
                dataset_preparer=dataset_preparer,
                mapping_layer=recommendation_cache.mapping_layer
            )
        )

        print("Preloaded recommendation system.")
        model_trained = True # Mark the model as trained
    except Exception as e:
        print(f"Error during model initialization: {e}")

# Lock to ensure thread safety when updating the cache
recommendation_cache_lock = threading.Lock()

# Function to train model at specified intervals, starting 1 hour before midnight IST
def periodic_model_training():
    global model_trained
    ist = pytz.timezone('Asia/Kolkata')
    interval_hours = 8  # Run every 8 hours

    while True:
        now = datetime.now(ist)

        # Calculate the next scheduled time (1 hour before midnight, and every 8 hours after that)
        midnight_ist = now.replace(hour=00, minute=10, second=0, microsecond=0)
        if now > midnight_ist:
            midnight_ist += timedelta(days=1)

        next_run_time = midnight_ist
        while next_run_time <= now:
            next_run_time += timedelta(hours=interval_hours)

        wait_seconds = (next_run_time - now).total_seconds()
        print(f"Next model training scheduled at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        time.sleep(wait_seconds)

        # Start model training
        try:
            print(f"Model training started at: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}")
            load_data()

            # Prepare datasets
            with recommendation_cache_lock:
                train_dataset = dataset_preparer.prepare_tf_dataset(
                    recommendation_cache.preprocessed_data,
                    recommendation_cache.mapping_layer
                )

            num_users = len(recommendation_cache.preprocessed_data['player_id'].unique())
            num_games = len(recommendation_cache.preprocessed_data['game_id'].unique())
            embedding_dim = 512

            siamese_model = SiameseRecommendationModel(
                num_users=num_users, num_games=num_games, embedding_dim=embedding_dim
            )
            model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
            model_trainer.train_model(train_dataset, train_dataset)  # Using train dataset for validation

            with recommendation_cache_lock:
                print("Updating recommendation system...")
                del recommendation_cache.recommendation_system  # Free memory
                recommendation_cache.update_recommendation_system(RecommendationSystem(
                    model=siamese_model,
                    dataset_preparer=dataset_preparer,
                    mapping_layer=recommendation_cache.mapping_layer
                ))

            print("Model retrained and recommendation system updated.")
            model_trained = True
        except Exception as e:
            print(f"Error during model training: {e}")

def compute_recommendations(request):
    """
    Generates a list of recommended users based on the input game and user details.
    
    If the model is not trained, it will trigger model training before generating recommendations.

    Parameters:
        request (object): An object containing the following attributes:
            - game_id (int): The ID of the game for which recommendations are generated.
            - user_id (str): The ID of the user requesting recommendations.
            - offset (int): The starting index for recommendations (used for pagination).
            - num_recommendations (int, optional): Number of recommendations to return (default handled by request).
            - filters (dict, optional): Filtering criteria such as country, expertise, and interests.

    Returns:
        dict: A dictionary containing game ID, user ID, and a list of recommended users.
    """
    global model_trained

    # Check if model is trained; if not, trigger model retraining
    if not model_trained:
        print("Model is not trained. Training now...")
        initialize_recommendation_system()
        
    return recommendation_cache.recommendation_system.recommend_top_users(
        recommendation_cache.preprocessed_data,
        request.game_id,
        request.user_id,
        request.offset,
        num_recommendations=request.num_recommendations,
        filters=request.filters
    )

def attach_usernames(top_users, user_map):
    """Adds `username` field to recommended users using the provided user mapping."""
    
    for user in top_users["recommended_users"]:
        user["username"] = user_map.get(user["user_id"], None)
        
def get_user_id_to_username_mapping(user_ids, name_field="full_name"):
    """Fetches user names in bulk for given user IDs and returns a mapping.

    Args:
        user_ids (list): List of user IDs.
        name_field (str): Field to use for name mapping, either 'full_name' or 'username'.

    Returns:
        dict: Mapping from user_id to specified name field.
    """
    
    if not user_ids:
        return {}

    user_data = fetch_all_users_data(
        table="user",
        database_name="crewdb", 
        columns=["id", name_field],
        conditions=[{"field": "id", "operator": "IN", "value": tuple(user_ids)}]
    )

    return {user["id"]: user[name_field] for user in user_data}
