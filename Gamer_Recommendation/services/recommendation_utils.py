from database import get_endpoint_data
from recommendation_system import RecommendationSystem
from data_preprocessing import DataPreprocessor, DatasetPreparer, MappingLayer
from models import SiameseRecommendationModel
from cache import RecommendationCache
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
def load_data():
    try:
        players_stats = get_endpoint_data().get('playersStats', {})
        RecommendationCache.preprocessed_data = preprocessor.preprocess(players_stats)
        RecommendationCache.mapping_layer = MappingLayer(RecommendationCache.preprocessed_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

# Function to train model at specified intervals, starting 1 hour before midnight IST
def periodic_model_training():
    global model_trained
    # Define IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    interval_hours = 8  # Run every 8 hours

    while True:
        # Get current time in IST
        now = datetime.now(ist)

        # Calculate the next scheduled time (1 hour before midnight, and every 8 hours after that)
        midnight_ist = now.replace(hour=23, minute=0, second=0, microsecond=0)
        if now > midnight_ist:  # If it's past 11 PM, calculate for the next day
            midnight_ist += timedelta(days=1)
        
        next_run_time = midnight_ist
        while next_run_time <= now:
            next_run_time += timedelta(hours=interval_hours)

        # Wait until the next run time
        wait_seconds = (next_run_time - now).total_seconds()
        print(f"Next model training scheduled at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        time.sleep(wait_seconds)

        # Start the model training process
        try:
            print(f"Model training started at: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}")
            load_data()

            # Prepare training and validation datasets
            train_dataset = dataset_preparer.prepare_tf_dataset(RecommendationCache.preprocessed_data,
                                                                RecommendationCache.mapping_layer)
            test_dataset = train_dataset  # For simplicity, using the same dataset for validation

            # Initialize and train a new model
            num_users = len(RecommendationCache.preprocessed_data['player_id'].unique())
            num_games = len(RecommendationCache.preprocessed_data['game_id'].unique())
            embedding_dim = 512
            siamese_model = SiameseRecommendationModel(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
            model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
            history = model_trainer.train_model(train_dataset, test_dataset)

            # Update the recommendation system with the new model
            RecommendationCache.recommendation_system = RecommendationSystem(
                model=siamese_model,
                dataset_preparer=dataset_preparer,
                mapping_layer=RecommendationCache.mapping_layer
            )

            print("Model retrained and recommendation system updated.")
            model_trained = True  # Mark the model as trained
        except Exception as e:
            print(f"Error during model training: {e}")

# Start periodic model training in a separate thread
threading.Thread(target=periodic_model_training, daemon=True).start()

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
        load_data()

        # Prepare training and validation datasets
        train_dataset = dataset_preparer.prepare_tf_dataset(RecommendationCache.preprocessed_data,
                                                            RecommendationCache.mapping_layer)
        test_dataset = train_dataset  # For simplicity, using the same dataset for validation

        # Initialize and train a new model
        num_users = len(RecommendationCache.preprocessed_data['player_id'].unique())
        num_games = len(RecommendationCache.preprocessed_data['game_id'].unique())
        embedding_dim = 512
        siamese_model = SiameseRecommendationModel(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
        model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
        history = model_trainer.train_model(train_dataset, test_dataset)

        # Update the recommendation system with the new model
        RecommendationCache.recommendation_system = RecommendationSystem(
            model=siamese_model,
            dataset_preparer=dataset_preparer,
            mapping_layer=RecommendationCache.mapping_layer
        )
        
        print("Model retrained and recommendation system updated.")
        model_trained = True  # Mark the model as trained

    return RecommendationCache.recommendation_system.recommend_top_users(
        RecommendationCache.preprocessed_data,
        request.game_id,
        request.user_id,
        request.offset,
        num_recommendations=request.num_recommendations,
        filters=request.filters
    )

def attach_usernames(top_users):
    """Adds a `username` field to each user in the recommendation list using their user ID."""
    from database import get_username  # Avoiding circular import by importing within function
    for user in top_users["recommended_users"]:
        user["username"] = get_username(user["user_id"])
