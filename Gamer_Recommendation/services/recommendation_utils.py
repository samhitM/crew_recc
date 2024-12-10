import tensorflow as tf
from core.database import get_endpoint_data
from recommendation_system.recommendation import (
    RecommendationSystem, DataPreprocessor, DatasetPreparer, MappingLayer, SiameseRecommendationModel
)
from recommendation_cache import recommendation_cache
from util.model_trainer import ModelTrainer
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
        recommendation_cache.preprocessed_data = preprocessor.preprocess(players_stats)
        recommendation_cache.mapping_layer = MappingLayer(recommendation_cache.preprocessed_data)
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
            train_dataset = dataset_preparer.prepare_tf_dataset(recommendation_cache.preprocessed_data,
                                                                recommendation_cache.mapping_layer)
            test_dataset = train_dataset  # For simplicity, using the same dataset for validation

            # Initialize and train a new model
            num_users = len(recommendation_cache.preprocessed_data['player_id'].unique())
            num_games = len(recommendation_cache.preprocessed_data['game_id'].unique())
            embedding_dim = 512
            siamese_model = SiameseRecommendationModel(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
            model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
            history = model_trainer.train_model(train_dataset, test_dataset)

            # Update the recommendation system with the new model
            recommendation_cache.recommendation_system = RecommendationSystem(
                model=siamese_model, vae_model=None,
                dataset_preparer=dataset_preparer,
                mapping_layer=recommendation_cache.mapping_layer
            )

            print("Model retrained and recommendation system updated.")
            model_trained = True  # Mark the model as trained
        except Exception as e:
            print(f"Error during model training: {e}")

# Start periodic model training in a separate thread
threading.Thread(target=periodic_model_training, daemon=True).start()

def compute_recommendations(request):
    """Generates a list of recommended users based on the input game and user details."""
    global model_trained

    # Check if model is trained; if not, trigger model retraining
    if not model_trained:
        print("Model is not trained. Training now...")
        load_data()

        # Prepare training and validation datasets
        train_dataset = dataset_preparer.prepare_tf_dataset(recommendation_cache.preprocessed_data,
                                                            recommendation_cache.mapping_layer)
        test_dataset = train_dataset  # For simplicity, using the same dataset for validation

        # Initialize and train a new model
        num_users = len(recommendation_cache.preprocessed_data['player_id'].unique())
        num_games = len(recommendation_cache.preprocessed_data['game_id'].unique())
        embedding_dim = 512
        siamese_model = SiameseRecommendationModel(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
        model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
        history = model_trainer.train_model(train_dataset, test_dataset)

        # Update the recommendation system with the new model
        recommendation_cache.recommendation_system = RecommendationSystem(
            model=siamese_model, vae_model=None,
            dataset_preparer=dataset_preparer,
            mapping_layer=recommendation_cache.mapping_layer
        )
        
        print("Model retrained and recommendation system updated.")
        model_trained = True  # Mark the model as trained

    return recommendation_cache.recommendation_system.recommend_top_users(
        recommendation_cache.preprocessed_data,
        request.game_id,
        request.user_id,
        request.offset,
        num_recommendations=request.num_recommendations,
        filters=request.filters
    )

def attach_usernames(top_users):
    """Adds a `username` field to each user in the recommendation list using their user ID."""
    from core.database import get_username  # Avoiding circular import by importing within function
    for user in top_users["recommended_users"]:
        user["username"] = get_username(user["user_id"])
