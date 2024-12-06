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

# Initialize components for recommendation system
preprocessor = DataPreprocessor()
dataset_preparer = DatasetPreparer()

# Function to load data and initialize recommendation system components
def load_data():
    try:
        players_stats = get_endpoint_data().get('playersStats', {})
        recommendation_cache.preprocessed_data = preprocessor.preprocess(players_stats)
        recommendation_cache.mapping_layer = MappingLayer(recommendation_cache.preprocessed_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

# Function to train model periodically and update recommendation cache
def periodic_model_training(interval=86400):  # Run every 24 hours by default
    while True:
        # Load fresh data
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

        # Wait for the next interval
        time.sleep(interval)

# Start periodic model training in a separate thread
threading.Thread(target=periodic_model_training, daemon=True).start()

def compute_recommendations(request):
    """Generates a list of recommended users based on the input game and user details."""
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
