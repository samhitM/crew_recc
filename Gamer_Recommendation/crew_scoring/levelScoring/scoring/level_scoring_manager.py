"""
Scoring calculations for level scoring.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
from database.level_db_manager import LevelDatabaseManager


class LevelScoringManager:
    """Manages all scoring calculations for level scoring."""
    
    def __init__(self):
        self.db_manager = LevelDatabaseManager()
        
        # Weights for final composite score
        self.composite_weights = {
            'gaming': 0.30,
            'impression': 0.25,
            'community': 0.10,
            'link_prediction': 0.20,
            'bonus': 0.15
        }
        
        # Gaming activity weights
        self.gaming_weights = {
            'max_hours': 0.4,
            'avg_hours': 0.35,
            'days_active': 0.25
        }
    
    def calculate_gaming_activity_score(self, gaming_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate gaming activity scores based on gaming time data."""
        print("Calculating gaming activity scores...")
        
        gaming_scores = {}
        
        for user_id, data in gaming_data.items():
            max_hours = data.get('max_hours', 0)
            avg_hours = data.get('avg_hours', 0)
            days_active = data.get('days_active', 0)
            
            # Calculate weighted gaming score
            gaming_score = (
                max_hours * self.gaming_weights['max_hours'] +
                avg_hours * self.gaming_weights['avg_hours'] +
                days_active * self.gaming_weights['days_active']
            )
            
            gaming_scores[user_id] = gaming_score
        
        print(f"Calculated gaming scores for {len(gaming_scores)} users")
        return gaming_scores
    
    def calculate_community_scores(self) -> Dict[str, float]:
        """Calculate community scores using graph-based approach."""
        print("Calculating community scores with link prediction...")
        
        try:
            # Try to detect communities
            print("Detecting communities with link prediction...")
            from graph.level_graph_manager import LevelGraphManager
            
            graph_manager = LevelGraphManager()
            graph_manager.build_community_graph()
            
            # Get link prediction scores
            link_prediction_scores = graph_manager.calculate_link_prediction_scores()
            
            # For now, use link prediction scores as community scores
            # Since community detection is failing, we'll use network centrality as proxy
            community_scores = {}
            
            if graph_manager.graph and graph_manager.graph.number_of_nodes() > 0:
                try:
                    # Calculate centrality measures as community proxy
                    degree_centrality = dict(graph_manager.graph.degree(weight='weight'))
                    
                    # Normalize degree centrality
                    if degree_centrality:
                        max_degree = max(degree_centrality.values())
                        if max_degree > 0:
                            community_scores = {node: degree / max_degree for node, degree in degree_centrality.items()}
                        else:
                            community_scores = {node: 0 for node in degree_centrality}
                    
                except Exception as e:
                    print(f"Error in community detection: {e}")
                    # Fall back to default values
                    community_scores = {node: 0 for node in graph_manager.graph.nodes()}
            
            print(f"Calculated community scores for {len(community_scores)} users")
            return community_scores
            
        except Exception as e:
            print(f"Error calculating community scores: {e}")
            return {}
    
    def calculate_bonus_factors(self, user_ids: List[str]) -> Dict[str, float]:
        """Calculate bonus factors for level scoring."""
        print("Calculating bonus factors...")
        
        bonus_scores = {}
        
        for user_id in user_ids:
            # Default bonus score
            bonus_score = 0.38  # Default value as used in original
            bonus_scores[user_id] = bonus_score
        
        print(f"Calculated bonus scores for {len(bonus_scores)} users")
        return bonus_scores
    
    def calculate_composite_scores_normalized(self, gaming_scores: Dict[str, float], 
                                            impression_scores: Dict[str, float],
                                            community_scores: Dict[str, float], 
                                            bonus_scores: Dict[str, float],
                                            link_prediction_scores: Dict[str, float]) -> tuple:
        """Calculate composite scores with comprehensive normalization."""
        print("Calculating composite scores with comprehensive normalization...")
        
        # Get all users
        all_users = set(gaming_scores.keys()) | set(impression_scores.keys()) | set(community_scores.keys()) | set(bonus_scores.keys()) | set(link_prediction_scores.keys())
        
        if not all_users:
            return {}, {}
        
        # Create a dataframe with all scores for normalization
        data = []
        for user_id in all_users:
            data.append({
                'user_id': user_id,
                'gaming_score': gaming_scores.get(user_id, 0),
                'impression_score': impression_scores.get(user_id, 0),
                'community_score': community_scores.get(user_id, 0),
                'bonus_score': bonus_scores.get(user_id, 0),
                'link_prediction_score': link_prediction_scores.get(user_id, 0)
            })
        
        df = pd.DataFrame(data)
        
        # Normalize all features comprehensively
        normalized_df = self.normalize_features_comprehensive(df)
        
        # Store normalized scores for output
        normalized_scores = {}
        for _, row in normalized_df.iterrows():
            user_id = row['user_id']
            normalized_scores[user_id] = {
                'norm_gaming_score': row['gaming_score'],
                'norm_impression_score': row['impression_score'],
                'norm_community_score': row['community_score'],
                'norm_bonus_score': row['bonus_score'],
                'norm_link_prediction_score': row['link_prediction_score']
            }
        
        # Calculate composite scores using normalized values
        composite_scores = {}
        for _, row in normalized_df.iterrows():
            user_id = row['user_id']
            composite_score = (
                row['gaming_score'] * self.composite_weights['gaming'] +
                row['impression_score'] * self.composite_weights['impression'] +
                row['community_score'] * self.composite_weights['community'] +
                row['link_prediction_score'] * self.composite_weights['link_prediction'] +
                row['bonus_score'] * self.composite_weights['bonus']
            )
            composite_scores[user_id] = composite_score
        
        print(f"Calculated normalized composite scores for {len(composite_scores)} users")
        return composite_scores, normalized_scores
    
    def normalize_features_comprehensive(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize all features to have mean=0 and std=1."""
        print("Normalizing level scoring features comprehensively (mean=0, std=1)...")
        
        # Create a copy to avoid modifying original
        normalized_df = data.copy()
        
        # Apply StandardScaler to each column
        scaler = StandardScaler()
        
        for column in normalized_df.columns:
            if column in ['user_id', 'crew_level']:  # Skip non-numeric columns
                continue
                
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
        
        numeric_cols = [col for col in normalized_df.columns if col not in ['user_id', 'crew_level']]
        print(f"Feature normalization completed. Mean values: {normalized_df[numeric_cols].mean().round(3).to_dict()}")
        print(f"Feature std values: {normalized_df[numeric_cols].std().round(3).to_dict()}")
        
        return normalized_df
