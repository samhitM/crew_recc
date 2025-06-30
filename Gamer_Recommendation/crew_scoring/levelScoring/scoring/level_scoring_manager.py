"""
Scoring calculations for level scoring.
"""
import pandas as pd
import networkx as nx
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
    
    def calculate_community_scores(self) -> tuple:
        """Calculate community scores using actual community detection."""
        print("Calculating community scores with actual community detection...")
        
        try:
            from graph.level_graph_manager import LevelGraphManager
            
            graph_manager = LevelGraphManager()
            graph_manager.build_community_graph()
            
            if not graph_manager.graph or graph_manager.graph.number_of_nodes() == 0:
                print("No graph available for community detection")
                return {}, {}
            
            # Convert to undirected graph for community detection
            undirected_graph = graph_manager.graph.to_undirected()
            
            # Detect communities using Louvain algorithm
            try:
                import networkx.algorithms.community as nx_comm
                communities = list(nx_comm.greedy_modularity_communities(undirected_graph, weight='weight'))
            except:
                # Fallback to simple connected components if Louvain fails
                communities = list(nx.connected_components(undirected_graph))
            
            print(f"Detected {len(communities)} communities")
            
            # Create community membership dictionary
            community_membership = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_membership[node] = i
            
            # Initialize community scores - all users start with 0
            community_scores = {node: 0.0 for node in graph_manager.graph.nodes()}
            
            # Get impression and gaming scores for ranking within communities
            impression_scores = self.get_impression_scores()
            gaming_data = self.db_manager.fetch_user_games_data()
            gaming_scores = self.calculate_gaming_activity_score(gaming_data)
            
            # Process each community
            for i, community in enumerate(communities):
                community_list = list(community)
                community_size = len(community_list)
                
                print(f"   Community {i+1}: {community_size} members")
                
                if community_size == 1:
                    # Single user communities get 0 score (not really a community)
                    continue
                
                # Give base community score (0.5) to all members
                for user in community_list:
                    community_scores[user] = 0.5
                
                # Calculate combined scores (impression + gaming) for ranking
                user_combined_scores = {}
                for user in community_list:
                    impression_score = impression_scores.get(user, 0)
                    gaming_score = gaming_scores.get(user, 0)
                    combined_score = impression_score + gaming_score
                    user_combined_scores[user] = combined_score
                
                # Sort users by combined score (descending)
                sorted_users = sorted(user_combined_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Give additional 0.4 points to top 5 performers in this community
                top_performers = min(5, len(sorted_users))
                for j in range(top_performers):
                    user_id = sorted_users[j][0]
                    community_scores[user_id] += 0.4
                    print(f"    Top performer {j+1}: User {user_id} (combined score: {sorted_users[j][1]:.3f}) -> community score: {community_scores[user_id]}")
            
            # Print summary
            members_count = sum(1 for score in community_scores.values() if score > 0)
            non_members_count = len(community_scores) - members_count
            top_performers_count = sum(1 for score in community_scores.values() if score > 0.5)
            
            print(f"Community scoring summary:")
            print(f"   Total users: {len(community_scores)}")
            print(f"   Community members: {members_count} (score: 0.5+)")
            print(f"   Non-members: {non_members_count} (score: 0.0)")
            print(f"   Top performers: {top_performers_count} (score: 0.9)")
            
            return community_scores, community_membership
            
        except Exception as e:
            print(f"Error in community detection: {e}")
            return {}, {}
    
    def get_impression_scores(self) -> Dict[str, float]:
        """Get normalized impression scores from impression scoring results."""
        print("Getting normalized impression scores...")
        
        # Try to load from the impression scoring results
        impression_file = "../impressionScoring/crew_impressions_revised.csv"
        
        try:
            import os
            if os.path.exists(impression_file):
                df = pd.read_csv(impression_file)
                # Filter out rows with empty user_ids
                df = df[df['user_id'].notna() & (df['user_id'] != '')]
                
                # Use normalized total impression score
                impression_scores = {}
                for _, row in df.iterrows():
                    user_id = row['user_id']
                    # Use the normalized total impression score
                    impression_score = row.get('norm_total_impression_score', 0)
                    impression_scores[user_id] = impression_score
                
                print(f"Loaded normalized impression scores for {len(impression_scores)} users from {impression_file}")
                return impression_scores
            else:
                print(f"Impression file not found: {impression_file}")
                # Try alternative paths
                alt_paths = [
                    "crew_impressions_revised.csv",
                    "../crew_impressions_revised.csv"
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        df = pd.read_csv(alt_path)
                        # Filter out rows with empty user_ids
                        df = df[df['user_id'].notna() & (df['user_id'] != '')]
                        
                        impression_scores = {}
                        for _, row in df.iterrows():
                            user_id = row['user_id']
                            impression_score = row.get('norm_total_impression_score', 0)
                            impression_scores[user_id] = impression_score
                        
                        print(f"Loaded impression scores from {alt_path}")
                        return impression_scores
                
                # If no file found, return empty dict
                print("No impression scores file found, using default values")
                return {}
                
        except Exception as e:
            print(f"Error loading impression scores: {e}")
            return {}
    
    def get_valid_impression_users(self) -> List[str]:
        """Get list of valid users from impression scoring results."""
        print("Getting valid users from impression data...")
        
        impression_file = "../impressionScoring/crew_impressions_revised.csv"
        
        try:
            import os
            if os.path.exists(impression_file):
                df = pd.read_csv(impression_file)
                # Filter out rows with empty user_ids
                df = df[df['user_id'].notna() & (df['user_id'] != '')]
                valid_users = df['user_id'].unique().tolist()
                print(f"Found {len(valid_users)} valid users in impression data")
                return valid_users
            else:
                print(f"Impression file not found: {impression_file}")
                return []
        except Exception as e:
            print(f"Error loading valid users: {e}")
            return []
    
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
