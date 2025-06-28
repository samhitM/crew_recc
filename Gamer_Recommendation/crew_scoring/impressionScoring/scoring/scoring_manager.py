"""
Scoring calculations for impression scoring.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from database.db_manager import DatabaseManager


class ScoringManager:
    """Manages all scoring calculations for impression scoring."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.scaler = StandardScaler()
        
        # Weights for topological score (pagerank + k_shell only)
        self.topological_weights = {
            'pagerank': 0.5,
            'k_shell': 0.5
        }
        
        # Default feature weights
        self.default_feature_weights = {
            'reposts': 0.15,
            'replies': 0.15,
            'mentions': 0.1,
            'favorites': 0.1,
            'interest_topic': 0.1,
            'bio_content': 0.05,
            'profile_likes': 0.1,
            'user_games': 0.1,
            'verified_status': 0.1,
            'posts_on_topic': 0.05,
            'messages': 0.1  # Added messages feature weight
        }
    
    def calculate_profile_likes(self, interaction_data: List[Dict]) -> Dict[str, int]:
        """Calculate profile likes from user interactions."""
        profile_likes = {}
        
        for interaction in interaction_data:
            entity_id = interaction.get('entity_id_primary')
            action = interaction.get('action', '').lower()
            interaction_type = interaction.get('interaction_type', '').upper()
            
            # Count likes for profile interactions
            if interaction_type == 'PROFILE_INTERACTION' and action == 'like':
                if entity_id:
                    profile_likes[entity_id] = profile_likes.get(entity_id, 0) + 1
        
        print(f"Calculated profile likes for {len(profile_likes)} users")
        return profile_likes
    
    def calculate_topological_score(self, graph_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate topological score combining PageRank and K-Shell only."""
        print("Calculating topological scores (PageRank + K-Shell)...")
        
        if not graph_metrics:
            return {}
        
        topological_scores = {}
        
        # Extract values for normalization
        pagerank_values = [metrics['pagerank'] for metrics in graph_metrics.values()]
        k_shell_values = [metrics['k_shell'] for metrics in graph_metrics.values()]
        
        # Normalize to [0, 1] range
        def normalize_values(values):
            if not values or max(values) == min(values):
                return [0.0] * len(values)
            return [(v - min(values)) / (max(values) - min(values)) for v in values]
        
        norm_pagerank = normalize_values(pagerank_values)
        norm_k_shell = normalize_values(k_shell_values)
        
        # Calculate weighted topological score (pagerank + k_shell)
        for i, user in enumerate(graph_metrics.keys()):
            topological_score = (
                self.topological_weights['pagerank'] * norm_pagerank[i] +
                self.topological_weights['k_shell'] * norm_k_shell[i]
            )
            topological_scores[user] = topological_score
        
        return topological_scores
    
    def prepare_feature_data(self, graph_metrics: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature data for linear regression."""
        print("Preparing feature data...")
        
        # Get gaming time data, message counts, and interaction data
        gaming_data = self.db_manager.fetch_user_games_data()
        message_data = self.db_manager.fetch_message_counts()
        interaction_data = self.db_manager.fetch_user_interactions()
        profile_likes_data = self.calculate_profile_likes(interaction_data)
        
        # Create feature matrix
        features = []
        user_ids = []
        
        for user_id, metrics in graph_metrics.items():
            # Feature values with actual gaming time, message counts, and profile likes
            feature_vector = {
                'reposts': 0,  # Default to 0 as specified
                'replies': 0,
                'mentions': 0,
                'favorites': 0,
                'interest_topic': 0,
                'bio_content': 0,
                'profile_likes': profile_likes_data.get(user_id, 0),  # Use actual profile likes
                'user_games': gaming_data.get(user_id, 0),  # Use actual gaming time
                'verified_status': 0,
                'posts_on_topic': 0,
                'messages': message_data.get(user_id, 0)  # Use actual message count
            }
            
            features.append(list(feature_vector.values()))
            user_ids.append(user_id)
        
        feature_names = list(feature_vector.keys())
        feature_df = pd.DataFrame(features, columns=feature_names, index=user_ids)
        
        # Normalize features comprehensively
        normalized_feature_df = self.normalize_features_comprehensive(feature_df)
        
        return normalized_feature_df, feature_names
    
    def learn_feature_weights(self, feature_df: pd.DataFrame, graph_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Learn feature weights using linear regression with PageRank as target."""
        print("Learning feature weights using linear regression...")
        
        try:
            # Prepare target (PageRank scores)
            pagerank_targets = [graph_metrics[user]['pagerank'] for user in feature_df.index]
            
            # Check if we have variation in features and targets
            if len(set(pagerank_targets)) <= 1:
                print("No variation in PageRank scores, using default weights")
                return self.default_feature_weights
            
            # Fit linear regression
            X = feature_df.values
            y = np.array(pagerank_targets)
            
            # Only fit if we have variation in features
            if np.var(X, axis=0).sum() == 0:
                print("No variation in features, using default weights")
                return self.default_feature_weights
            
            lr = LinearRegression()
            lr.fit(X, y)
            
            # Extract weights
            feature_weights = {}
            for i, feature_name in enumerate(feature_df.columns):
                feature_weights[feature_name] = max(0, lr.coef_[i])  # Ensure non-negative weights
            
            # Normalize weights to sum to 1
            total_weight = sum(feature_weights.values())
            if total_weight > 0:
                feature_weights = {k: v/total_weight for k, v in feature_weights.items()}
            else:
                feature_weights = self.default_feature_weights
            
            print("Learned feature weights:", feature_weights)
            return feature_weights
            
        except Exception as e:
            print(f"Error in learning feature weights: {e}")
            print("Using default feature weights")
            return self.default_feature_weights
    
    def calculate_user_feature_scores(self, feature_df: pd.DataFrame, feature_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate user feature scores using learned weights."""
        print("Calculating user feature scores...")
        
        feature_scores = {}
        
        for user_id in feature_df.index:
            score = 0
            for feature_name, weight in feature_weights.items():
                if feature_name in feature_df.columns:
                    score += feature_df.loc[user_id, feature_name] * weight
            feature_scores[user_id] = score
        
        return feature_scores
    
    def calculate_website_impressions(self, user_ids: List[str]) -> Dict[str, float]:
        """Calculate website impressions based on profile visits."""
        print("Calculating website impressions...")
        
        # Default to 0 
        website_impressions = {}
        
        for user_id in user_ids:
            # Default formula: unique_pageviews * (1 + scroll_depth_percent/100)
            unique_pageviews = 0  # Default to 0
            scroll_depth_percent = 0  # Default to 0
            
            impression_score = unique_pageviews * (1 + scroll_depth_percent / 100)
            website_impressions[user_id] = impression_score
        
        return website_impressions
    
    def normalize_features_comprehensive(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all features to have mean=0 and std=1."""
        print("Normalizing features comprehensively (mean=0, std=1)...")
        
        # Create a copy to avoid modifying original
        normalized_df = feature_df.copy()
        
        # Apply StandardScaler to each column
        scaler = StandardScaler()
        
        for column in normalized_df.columns:
            # Get column values
            values = normalized_df[column].values.reshape(-1, 1)
            
            # Check if there's variation in the column
            if normalized_df[column].std() > 0:
                # Normalize to mean=0, std=1
                normalized_values = scaler.fit_transform(values).flatten()
                normalized_df[column] = normalized_values
            else:
                # If no variation, set all values to 0
                normalized_df[column] = 0.0
        
        print(f"Feature normalization completed. Mean values: {normalized_df.mean().round(3).to_dict()}")
        print(f"Feature std values: {normalized_df.std().round(3).to_dict()}")
        
        return normalized_df
