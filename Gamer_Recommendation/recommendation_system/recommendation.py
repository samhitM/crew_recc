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
from database import init_db_pools
from datetime import datetime

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
        
    def _empty_response(self, game_id, user_id, offset):
        return {
            "game_id": int(game_id),
            "user_id":  str(user_id),
            "offset": int(offset),
            "user_specialization": [],
            "recommended_users": []
        }
    
    def calculate_age(self, dob):
        if dob is None:
            return None
        today = datetime.today()
        # Assuming dob is in 'YYYY-MM-DD' format
        birth_date = datetime.strptime(str(dob), '%Y-%m-%d')
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age

    def recommend_top_users(self, df, game_id, user_id, offset, num_recommendations=20, filters=None):
        from database.queries import fetch_all_users_data
        try:
            if df.empty:
                return self._empty_response(game_id, user_id, offset)

            original_user_id = user_id
            fallback_user_id = "dCUKB2Vf9Zk"

            user_age_row = df[df['player_id'] == user_id]
            if user_age_row.empty:
                user_id = fallback_user_id
                dob_records = fetch_all_users_data(
                    table="user",
                    database_name="crewdb", 
                    columns=["dob"],
                    conditions=[{"field": "id", "operator": "=", "value": user_id}],
                    limit=1
                )
                if not dob_records:
                    return self._empty_response(game_id, original_user_id, offset)
                user_dob = self.calculate_age(dob_records[0]['dob'])
            else:
                user_dob = user_age_row['age'].values[0]

            all_specializations = get_specialisations(df['player_id'].tolist())
            user_specializations = all_specializations.get(original_user_id, [])

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
                return self._empty_response(game_id, original_user_id, offset)
            
            api_client = APIClient(API_BASE_URL, original_user_id)
            relationships = {
                "friends": set(relation['player_id'] for relation in api_client.fetch_user_relations(limit=50, offset=0, relation="friends")),
                "blocked": set(relation['player_id'] for relation in api_client.fetch_user_relations(limit=50, offset=0, relation="blocked_list")),
                "reported": set(relation['player_id'] for relation in api_client.fetch_user_relations(limit=50, offset=0, relation="report_list"))
            }

            df = df[~df['player_id'].isin(relationships["blocked"] | relationships["reported"])]

            game_idx = self.mapping_layer.map_game_id(game_id)
            user_idx = self.mapping_layer.map_user_id(user_id)
            if game_idx is None or user_idx is None:
                raise ValueError("Invalid game_id or user_id mapping.")

            user_input_indices = df['player_id'].map(self.mapping_layer.user_id_mapping).values.astype(np.int32)
            game_input = np.full(len(user_input_indices), game_idx, dtype=np.int32)

            (_, _, game_features, global_features), _ = self.dataset_preparer.create_input_tensors(df, self.mapping_layer)

            # Safety: check again right before predict
            if len(user_input_indices) == 0:
                return self._empty_response(game_id, original_user_id, offset)

            inputs = (user_input_indices, game_input, game_features, global_features)
            scores = self.model.predict(inputs)

            df['score'] = scores
            min_score = df['score'].min()
            if min_score < 0:
                df['score'] -= min_score

            interaction_map = get_interaction_type(original_user_id, df['player_id'].tolist())
            df = ScoreAdjuster.adjust_scores(df, interaction_map, relationships)

            df = df[df['player_id'] != original_user_id]
            if df.empty:
                return self._empty_response(game_id, original_user_id, offset)

            top_users = df.nlargest(num_recommendations, 'score')
            recommended_users = [
                {
                    "user_id": row['player_id'],
                    "country": row['country'],
                    "age": int(row['age']),
                    "recommendation_expertise": row['recommendation_expertise'],
                    "interests": row['user_interests'],
                    "recommendation_specialization": all_specializations.get(row['player_id'], [])
                }
                for _, row in top_users.iterrows()
            ]

            return {
                "game_id": int(game_id),
                "user_id": str(original_user_id),
                "offset": int(offset),
                "user_specialization": user_specializations,
                "recommended_users": recommended_users,
            }

        except Exception as e:
            print(f"Error in recommend_top_users: {e}")
            return self._empty_response(game_id, user_id, offset)
    
    
if __name__ == "__main__":
    # Load your data and initialize the model, dataset preparer, and mapping layer here
    init_db_pools()
    
    data_loader = DataLoader('../Datasets/generated_players.json')
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

    model_trainer = ModelTrainer(model=siamese_model, learning_rate=0.001, batch_size=128, epochs=15)
    train_dataset = dataset_preparer.prepare_tf_dataset(preprocessed_data, mapping_layer)
    test_dataset = train_dataset  # Replace with actual test dataset
    history = model_trainer.train_model(train_dataset, test_dataset)
    
    # siamese_model.save('../Gamer_Recommendation/models/siamese_model.keras')
    
    LossPlotter.plot_loss(history)

    recommendation_system = RecommendationSystem(
        model=siamese_model, dataset_preparer=dataset_preparer, mapping_layer=mapping_layer)

    
    game_id= 578080
    user_id= "8qqQdeMC3s5"
    offset=10
    num_recommendations=10
    filters= {
        "recommendation_expertise": "beginner",
        "recommendation_specialisation": "Strategy",
        "country": "India",
        "user_interests": ["Action", "MOBA", "Strategy", "Indie", "RPG"],
        "age_delta": 35
    }
    

    result = recommendation_system.recommend_top_users(
        df=preprocessed_data, game_id=game_id, user_id=user_id,offset=offset,num_recommendations=num_recommendations, filters=filters)
    
    print(result)
    # print(json.dumps(result, indent=2))