import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from database import get_interaction_type, get_specialisations
from utils import DataFilter
from utils import ScoreAdjuster
from utils import APIClient
from utils import LossPlotter
from utils import ModelTrainer

# Import custom modules from the respective directories
from data_preprocessing import DataLoader
from data_preprocessing import DataPreprocessor
from data_preprocessing import MappingLayer
from data_preprocessing import DatasetPreparer
from models import SiameseRecommendationModel
from core.config import API_BASE_URL


class RecommendationSystem:
    def __init__(self, model, dataset_preparer, mapping_layer):
        self.model = model
        self.dataset_preparer = dataset_preparer
        self.mapping_layer = mapping_layer

    def recommend_top_users(self, df, game_id, user_id, offset, num_recommendations=20, filters=None):
        """
        Generates a list of recommended users based on various filters and a predictive model.

        Parameters:
            df (DataFrame): User dataset containing player information.
            game_id (int): The ID of the game for which recommendations are generated.
            user_id (str): The ID of the requesting user.
            offset (int): The starting index for recommendations (used for pagination).
            num_recommendations (int, optional): The number of users to recommend (default: 20).
            filters (dict, optional): Additional filtering criteria such as country, expertise, and interests.

        Returns:
            dict: A dictionary containing the game ID, user ID, and a list of recommended users.
        """
    
        try:
            user_dob = df[df['player_id'] == user_id]['age'].values[0]
            user_specializations = get_specialisations(user_id=user_id)  
            
            if filters:
                 df = DataFilter.process_filters(
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
            
            api_client = APIClient(API_BASE_URL,user_id)
            
            # Fetch user relationships in a single batch
            relationships = {
                "friends": set(relation['player_id'] for relation in api_client.fetch_user_relations(limit=50, offset=0, relation="friends")),
                "blocked": set(relation['player_id'] for relation in api_client.fetch_user_relations(limit=50, offset=0, relation="blocked_list")),
                "reported": set(relation['player_id'] for relation in api_client.fetch_user_relations(limit=50, offset=0, relation="report_list"))
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
            df['score'] = df.apply(lambda row: ScoreAdjuster.adjust_score(row, interaction_map, relationships), axis=1)

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
    
    LossPlotter.plot_loss(history)

    recommendation_system = RecommendationSystem(
        model=siamese_model, dataset_preparer=dataset_preparer, mapping_layer=mapping_layer)

    
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