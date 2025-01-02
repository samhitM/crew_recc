import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json
import matplotlib.pyplot as plt
from datetime import datetime
import requests

# Import custom modules from the respective directories
from data_preprocessing.data_loader import DataLoader
from data_preprocessing.data_preprocessor import DataPreprocessor
from data_preprocessing.mapping_layer import MappingLayer
from data_preprocessing.dataset_preparer import DatasetPreparer
from models.siamese_recommendation_model import SiameseRecommendationModel
from util.model_trainer import ModelTrainer
from core.database import get_interaction_type,get_specialisations

import warnings
from requests.exceptions import RequestException

from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RecommendationSystem:
    def __init__(self, model, vae_model, dataset_preparer, mapping_layer):
        self.model = model
        self.vae_model = vae_model
        self.dataset_preparer = dataset_preparer
        self.mapping_layer = mapping_layer

    def fetch_user_relations(self, jwt_token=None, limit=50, offset=0, relation=None):
        """Fetch user relations with optional pagination and filtering."""
        
        jwt_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4UHpJOG5FUXU1TCIsImVtYWlsIjoieWFzaHlhZGF2MDBAZmxhc2guY28iLCJpYXQiOjE3MzIxOTE3MjYsImV4cCI6MTczNzM3NTcyNn0.DeRiGzNUflr6_8CSqrw3K7UkybEb8pJe9ocD9Gs5Axs"
        
        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }
        # Set up the query parameters
        params = {
            'limit': limit,
            'offset': offset,
        }
        
        # Request body for the relation filter
        data = {}
        if relation:
            data['relation'] = relation
        
        try:
            # Send the POST request with headers, query parameters, and request body
            response = requests.get(
                'https://localhost:3000/api/user/relations', ## Modify this when server gets hosted
                headers=headers, 
                params=params,
                json=data,
                verify=False
            )

            if response.status_code == 200:
                # Parse the JSON response
                relations = response.json()
                return [
                    {
                        "player_id": r["user_id"],
                        "relation": relation,
                        "user_interests": r.get("user_interests", []),
                        "played_games": r.get("played_games", []),
                        "last_active_ts": r.get("last_active_ts", ""),
                    }
                    for r in relations
                ]
            elif response.status_code == 404:
                # Handle no relations found
                return []
            else:
                # Raise an error for unexpected status codes
                raise ValueError(f"Failed to fetch relations. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"An error occurred while fetching relations: {str(e)}")
        

    def process_filters(self, df, game_id, country, recommendation_expertise, user_interests, age, delta=None):
        """Filter the dataframe based on game_id and other filters, with case-insensitive comparisons."""
        # Filter by game_id (no change needed for game_id)
        filtered_df = df[df['game_id'] == game_id]
        if filtered_df.empty:
            return filtered_df  # Return empty DataFrame early

        # Filter by country (case-insensitive)
        if country:
            filtered_df = filtered_df[filtered_df['country'].str.lower() == country.lower()]
            if filtered_df.empty:
                return filtered_df

        # # Filter by recommendation_expertise (case-insensitive)
        if recommendation_expertise:
            filtered_df = filtered_df[filtered_df['recommendation_expertise'].str.lower() == recommendation_expertise.lower()]
            if filtered_df.empty:
                return filtered_df

        # # Filter by user_interests (case-insensitive)
        if user_interests:
            user_interests = [interest.lower() for interest in user_interests]  # Convert user interests list to lowercase
            filtered_df = filtered_df[
                filtered_df['user_interests'].apply(lambda x: any(interest in [i.lower() for i in x] for interest in user_interests))
            ]
            if filtered_df.empty:
                return filtered_df
            
        # Filter by DOB with delta (no change needed for numeric values)
        if delta:
            filtered_df = filtered_df[
                (filtered_df['age'] >= age - delta) & (filtered_df['age'] <= age + delta)
            ]
            if filtered_df.empty:
                return filtered_df

        return filtered_df


    def recommend_top_users(self, df, game_id, user_id, offset, num_recommendations=20, filters=None, jwt_token=None):
        try:
            user_dob = df[df['player_id'] == user_id]['age'].values[0]
            user_specializations = get_specialisations(user_id=user_id)  

            # Apply filters if provided
            if filters:
                df = self.process_filters(
                    df,
                    game_id,
                    filters.get('country'),
                    filters.get('recommendation_expertise'),
                    filters.get('user_interests'),
                    user_dob,
                    filters.get('age_delta', 12)
                )
            if df.empty:
                return {"game_id": game_id, "user_id": user_id, "recommended_users": []}
            
            # Fetch user relationships in a single batch
            relationships = {
                "friends": set(relation['player_id'] for relation in self.fetch_user_relations(jwt_token=jwt_token, limit=50, offset=0, relation="friends")),
                "blocked": set(relation['player_id'] for relation in self.fetch_user_relations(jwt_token=jwt_token, limit=50, offset=0, relation="blocked_list")),
                "reported": set(relation['player_id'] for relation in self.fetch_user_relations(jwt_token=jwt_token, limit=50, offset=0, relation="report_list"))
            }
            
            # Exclude blocked and reported users early
            df = df[~df['player_id'].isin(relationships["blocked"] | relationships["reported"])]
            if df.empty:
                return {"game_id": game_id, "user_id": user_id, "recommended_users": []}
            

            # Map game_id and user_id to indices
            game_idx = self.mapping_layer.map_game_id(game_id)
            user_idx = self.mapping_layer.map_user_id(user_id)
            if game_idx is None or user_idx is None:
                raise ValueError("Invalid game_id or user_id mapping.")

            # Prepare inputs for model prediction
            user_input_indices = df['player_id'].map(self.mapping_layer.user_id_mapping).values.astype(np.int32)
            game_input = np.full(len(user_input_indices), game_idx, dtype=np.int32)

            # Extract features and predict scores
            (_, _, game_features, global_features), _ = self.dataset_preparer.create_input_tensors(df, self.mapping_layer)
            inputs = (user_input_indices, game_input, game_features, global_features)
            scores = self.model.predict(inputs)

            # Aggregate scores using original player IDs
            df['score'] = scores
            min_score = df['score'].min()
            if min_score < 0:
                df['score'] -= min_score  # Normalize scores to non-negative

            # Pre-fetch interactions and specializations for optimization
            interaction_map = {row['player_id']: get_interaction_type(user_id, row['player_id']) for _, row in df.iterrows()}
            specializations_map = {row['player_id']: get_specialisations(row['player_id']) for _, row in df.iterrows()}
            
            # Adjust scores for interactions
            def adjust_score(row):
                score = row['score']
                interaction = interaction_map.get(row['player_id'])
                if interaction:
                    if interaction['eventType'] == "PROFILE_INTERACTION" and interaction['action'] == "friend_request":
                        score *= 0.5
                    elif interaction['eventType'] == "PROFILE_INTERACTION" and interaction['action'] == "ignored":
                        time_elapsed = (datetime.now() - interaction['createTimestamp']).days
                        decay_factor = max(0, 1 - time_elapsed / 30)
                        score *= decay_factor

                # Adjust for friends
                if row['player_id'] in relationships["friends"]:
                    score *= 0.8
                return score

            df['score'] = df.apply(adjust_score, axis=1)

            # Exclude the requesting user and sort by score
            df = df[df['player_id'] != user_id]
            top_users = df.nlargest(num_recommendations, 'score')

            # Prepare detailed response
            recommended_users = [
                {
                    "user_id": row['player_id'],
                    "country": row['country'],
                    "age": int(row['age']),
                    "recommendation_expertise": row['recommendation_expertise'],
                    "interests": row['user_interests'],
                    "recommendation_specialization": specializations_map.get(row['player_id'], [])
                }
                for _, row in top_users.iterrows()
            ]

            # Final response
            response = {
                "game_id": int(game_id),
                "user_id": str(user_id),
                "offset": int(offset),
                "user_specialization": user_specializations,
                "recommended_users": recommended_users,
            }

            return response

        except Exception as e:
            raise ValueError(f"An error occurred while generating recommendations: {str(e)}")


def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load your data and initialize the model, dataset preparer, and mapping layer here
    data_loader = DataLoader('../Datasets/data.json')
    raw_data = data_loader.load_data()

    preprocessor = DataPreprocessor()
    num_recommendations = 3
    players_stats = raw_data['playersStats']
    preprocessed_data = preprocessor.preprocess(players_stats)
    
    mapping_layer = MappingLayer(preprocessed_data)
    
    num_users = len(preprocessed_data['player_id'].unique())
    num_games = len(preprocessed_data['game_id'].unique())
    embedding_dim = 512

    siamese_model = SiameseRecommendationModel(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
    vae_model = None  # Replace with the actual VAE model (Will integrate in the second version)
    dataset_preparer = DatasetPreparer()

    model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=65)
    train_dataset = dataset_preparer.prepare_tf_dataset(preprocessed_data, mapping_layer)
    test_dataset = train_dataset  # Replace with actual test dataset
    history = model_trainer.train_model(train_dataset, test_dataset)
    
    # siamese_model.save('../Gamer_Recommendation/models/siamese_model.keras')
    
    plot_loss(history)

    recommendation_system = RecommendationSystem(
        model=siamese_model, vae_model=vae_model, dataset_preparer=dataset_preparer, mapping_layer=mapping_layer)

    
    game_id= 578080
    user_id= "dCUKB2Vf9Zk"
    offset=10
    num_recommendations=2
    filters= {
        "recommendation_expertise": "beginner",
        "recommendation_specialisation": "Strategy",
        "country": "India",
        "user_interests": ["Action", "MOBA", "Strategy", "Indie", "RPG", "New Interest 1"],
        "age_delta": 7
    }
    

    result = recommendation_system.recommend_top_users(
        df=preprocessed_data, game_id=game_id, user_id=user_id,offset=offset,num_recommendations=num_recommendations, filters=filters)
    
    print(result)
    # print(json.dumps(result, indent=2))

