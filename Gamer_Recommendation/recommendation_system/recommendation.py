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
        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }
        
        # Set up the query parameters
        params = {
            'limit': limit,
            'offset': offset,
        }
        
        if relation:
            params['relation'] = relation
        
        # Send the GET request with headers and query parameters
        response = requests.get('http://localhost/user/relations', headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Failed to fetch relations. Status code: {response.status_code}")

    
    def fetch_user_specializations(self, user_id, jwt_token=None):
        return []
        """Fetch user specializations based on user_id and JWT token."""
        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }
        response = requests.get(f'http://localhost/api/specializations/{user_id}', headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            # If no specializations exist, return an empty list
            return []

    def process_filters(self, df, game_id, country, recommendation_expertise, user_interests,age, delta=None):
        """Filter the dataframe based on game_id and other filters."""
        # Filter by game_id
        filtered_df = df[df['game_id'] == game_id]
        if filtered_df.empty:
            return filtered_df  # Return empty DataFrame early

        # Filter by country
        if country:
            filtered_df = filtered_df[filtered_df['country'] == country]
            if filtered_df.empty:
                return filtered_df

        # Filter by recommendation_expertise
        if recommendation_expertise:
            filtered_df = filtered_df[filtered_df['recommendation_expertise'] == recommendation_expertise]
            if filtered_df.empty:
                return filtered_df

        # Filter by user_interests
        if user_interests:
            filtered_df = filtered_df[
                filtered_df['user_interests'].apply(lambda x: any(interest in x for interest in user_interests))
            ]
            if filtered_df.empty:
                return filtered_df

        # Filter by DOB with delta
        if delta:
            filtered_df = filtered_df[
                (filtered_df['age'] >= age-delta) & (filtered_df['age'] <= age+delta)
            ]
            if filtered_df.empty:
                return filtered_df

        return filtered_df

    def recommend_top_users(self, df, game_id, user_id,offset,num_recommendations=20, filters=None,jwt_token=None):
        try:
            print(game_id,user_id,offset,num_recommendations,filters)
            user_dob = df[df['player_id'] == user_id]['age'].values[0] 
            # Apply filters if provided
            if filters:
                df = self.process_filters(
                    df,
                    game_id,
                    filters.get('country'),
                    filters.get('recommendation_expertise'),
                    filters.get('user_interests'),
                    user_dob,
                    filters.get('age_delta',12)
                )
            if df.empty:
                return {"game_id": game_id, "user_id": user_id, "recommended_users": []}

            # Fetch relations for the user
            #relations = self.fetch_user_relations(user_id, jwt_token=jwt_token)
            relations={}
            blocked_users = [r['user_id'] for r in relations if r.get('relation') == 'blocked_list']
            reported_users = [r['user_id'] for r in relations if r.get('relation') == 'reported_list']
            friends = [r['user_id'] for r in relations if r.get('relation') == 'friends']

            # Fetch specializations for the user
            specializations = self.fetch_user_specializations(user_id, jwt_token=jwt_token)

            # Filter out blocked and reported users
            df = df[~df['player_id'].isin(blocked_users + reported_users)]
            
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

            # Extract game and global features using dataset preparer
            (_, _, game_features, global_features), _ = self.dataset_preparer.create_input_tensors(
                df, self.mapping_layer
            )

            # Predict compatibility scores
            inputs = (user_input_indices, game_input, game_features, global_features)
            scores = self.model.predict(inputs)
            

            # Aggregate scores using original player IDs
            user_scores = {}
            for idx, score in enumerate(scores):
                original_user_id = int(df.iloc[idx]['player_id'])  # Ensure native int type
                user_scores[original_user_id] = user_scores.get(original_user_id, 0) + float(score)

            # Normalize scores to make all values non-negative
            min_score = min(user_scores.values())
            if min_score < 0:
                user_scores = {user: score - min_score for user, score in user_scores.items()}

            # Adjust scores for friends (lower their priority)
            for friend_id in friends:
                if friend_id in user_scores:
                    user_scores[friend_id] *= 0.8  # Reduce the score by 20% for friends

            # Adjust scores for specializations (boost users with similar specializations)
            for spec in specializations:
                df_with_spec = df[df['specialization'].apply(lambda x: spec in x)]
                for idx, row in df_with_spec.iterrows():
                    player_id = int(row['player_id'])
                    if player_id in user_scores:
                        user_scores[player_id] *= 1.2  # Increase score by 20% for matching specializations

            # Exclude the requesting user from recommendations
            if user_id in user_scores:
                user_scores[user_id] = min(user_scores.values()) - 1  # Lower the requesting user's score

            # Sort users by scores in descending order
            sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

            # Limit to the top recommendations
            top_user_ids = [str(user_id) for user_id, _ in sorted_users][:num_recommendations]

            # Prepare detailed response with user info
            recommended_users = []
            for uid in top_user_ids:
                user_info = df[df['player_id'] == uid].iloc[0]
                recommended_users.append({
                    "user_id": int(user_info['player_id']),
                    #"score": round(user_scores[uid], 2),
                    "country": user_info['country'],
                    "age": int(user_info['age']),
                    "recommendation_expertise": user_info['recommendation_expertise'],
                    "interests": user_info['user_interests'],
                    #"specialization": user_info['specialization'],  # Include specialization in the response
                    "user_specialization": [],
                    "recommendation_specialization": []
                })

            # Prepare the response object
            response = {
                "game_id": int(game_id),
                "user_id": str(user_id),
                "offset": int(offset),
                "recommended_users": recommended_users
            }

            # Write the response to a JSON file
            # with open('response.json', 'w') as json_file:
            #     json.dump(response, json_file, indent=4)

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
    data_loader = DataLoader('Datasets/data.json')
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
    
    siamese_model.save('../Gamer_Recommendation/models/siamese_model.keras')
    
    plot_loss(history)

    recommendation_system = RecommendationSystem(
        model=siamese_model, vae_model=vae_model, dataset_preparer=dataset_preparer, mapping_layer=mapping_layer)

    
    game_id= 252950
    user_id= "820BKNsM0PI"
    offset=10
    num_recommendations=2
    filters= {
        "friend_type": "Pro",
        "country": "Germany",
        "user_interests": ["Action", "MOBA", "Strategy", "Indie", "RPG"]
    }
    

    result = recommendation_system.recommend_top_users(
        df=preprocessed_data, game_id=game_id, user_id=user_id,offset=offset,num_recommendations=num_recommendations, filters=filters)
    
    print(json.dumps(result, indent=2))

