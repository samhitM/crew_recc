"""
Main level scoring calculator - modularized version.
"""
import os
import pandas as pd
from typing import Dict
import warnings

from database.level_db_manager import LevelDatabaseManager
from graph.level_graph_manager import LevelGraphManager
from scoring.level_scoring_manager import LevelScoringManager
from clustering.level_clustering_manager import LevelClusteringManager
from utils.level_helpers import LevelFileUtils, LevelValidationUtils

warnings.filterwarnings("ignore")


class LevelCalculator:
    """
    Modularized implementation for calculating crew levels.
    """
    
    def __init__(self):
        self.db_manager = LevelDatabaseManager()
        self.graph_manager = LevelGraphManager()
        self.scoring_manager = LevelScoringManager()
        self.clustering_manager = LevelClusteringManager(target_levels=3)
        self.file_utils = LevelFileUtils()
        self.validation_utils = LevelValidationUtils()
    
    def calculate_crew_levels(self) -> pd.DataFrame:
        """Main method to calculate crew levels."""
        print("Starting revised crew level calculation...")
        
        # Check if results already exist
        output_file = "crew_levels_revised.csv"
        if os.path.exists(output_file):
            print(f"Found existing level results in {output_file}")
            try:
                existing_df = pd.read_csv(output_file)
                print(f"Loaded existing level data for {len(existing_df)} users")
                return existing_df
            except Exception as e:
                print(f"Error reading existing file: {e}")
                print("Proceeding with fresh calculation...")
        
        # Step 1: Get gaming data and calculate gaming scores
        gaming_data = self.db_manager.fetch_user_games_data()
        gaming_scores = self.scoring_manager.calculate_gaming_activity_score(gaming_data)
        
        # Step 2: Get impression scores
        impression_scores = self.db_manager.get_impression_scores()
        
        # Step 3: Calculate community scores
        community_scores = self.scoring_manager.calculate_community_scores()
        
        # Step 4: Calculate link prediction scores
        link_prediction_scores = self.graph_manager.calculate_link_prediction_scores()
        
        # Visualize the community graph
        self.graph_manager.visualize_community_graph("community_graph_with_interactions.png")
        
        # Step 5: Calculate bonus factors
        all_users = set(gaming_scores.keys()) | set(impression_scores.keys()) | set(community_scores.keys()) | set(link_prediction_scores.keys())
        bonus_scores = self.scoring_manager.calculate_bonus_factors(list(all_users))
        
        # Step 6: Calculate composite scores with normalization
        composite_scores, normalized_scores = self.scoring_manager.calculate_composite_scores_normalized(
            gaming_scores, impression_scores, community_scores, bonus_scores, link_prediction_scores
        )
        
        # Step 7: Use KNN clustering to assign levels
        level_assignments = self.clustering_manager.assign_levels_with_knn_clustering(composite_scores)
        
        # Step 8: Validate level distribution
        self.validation_utils.validate_level_distribution(level_assignments)
        
        # Step 9: Create results dataframe
        results = []
        for user_id in all_users:
            # Get normalized scores for this user
            norm_scores = normalized_scores.get(user_id, {})
            
            results.append({
                'user_id': user_id,
                'gaming_score': gaming_scores.get(user_id, 0),
                'impression_score': impression_scores.get(user_id, 0),
                'community_score': community_scores.get(user_id, 0),
                'link_prediction_score': link_prediction_scores.get(user_id, 0),
                'bonus_score': bonus_scores.get(user_id, 0),
                'composite_score': composite_scores.get(user_id, 0),
                'crew_level': level_assignments.get(user_id, 1),
                'gaming_time': gaming_data.get(user_id, {}).get('max_hours', 0),
                # Add normalized scores
                'norm_gaming_score': norm_scores.get('norm_gaming_score', 0),
                'norm_impression_score': norm_scores.get('norm_impression_score', 0),
                'norm_community_score': norm_scores.get('norm_community_score', 0),
                'norm_link_prediction_score': norm_scores.get('norm_link_prediction_score', 0),
                'norm_bonus_score': norm_scores.get('norm_bonus_score', 0),
                'norm_composite_score': composite_scores.get(user_id, 0)  # Composite score is already normalized
            })
        
        df = pd.DataFrame(results)
        print(f"Calculated crew levels for {len(df)} users")
        
        # Save results with error handling
        self.file_utils.save_results_with_fallback(df, output_file)
        
        return df


if __name__ == "__main__":
    calculator = LevelCalculator()
    results_df = calculator.calculate_crew_levels()
    
    if not results_df.empty:
        print(f"Successfully calculated levels for {len(results_df)} users")
    else:
        print("No results to save")
