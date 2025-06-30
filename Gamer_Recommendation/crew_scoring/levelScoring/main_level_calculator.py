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
from utils.level_helpers import LevelFileUtils, LevelValidationUtils, LevelAggregationUtils, LevelVisualizationUtils

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
        self.aggregation_utils = LevelAggregationUtils()
        self.visualization_utils = LevelVisualizationUtils()
    
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
        
        # Step 0: Get valid users from impression data
        valid_impression_users = self.scoring_manager.get_valid_impression_users()
        if not valid_impression_users:
            print("No valid users found in impression data. Cannot proceed.")
            return pd.DataFrame()
        
        print(f"Will calculate levels for {len(valid_impression_users)} users from impression data")
        
        # Step 1: Get gaming data and calculate gaming scores
        gaming_data = self.db_manager.fetch_user_games_data()
        # Filter gaming data to only include users from impression data
        gaming_data = {user_id: data for user_id, data in gaming_data.items() 
                      if user_id in valid_impression_users}
        gaming_scores = self.scoring_manager.calculate_gaming_activity_score(gaming_data)
        
        # Step 2: Get impression scores
        impression_scores = self.db_manager.get_impression_scores()
        
        # Step 3: Calculate community scores and get community membership
        community_scores, community_membership = self.scoring_manager.calculate_community_scores()
        
        # Step 4: Calculate link prediction scores
        link_prediction_scores = self.graph_manager.calculate_link_prediction_scores()
        
        # Visualize the community graph with community circles
        self.graph_manager.visualize_community_graph("community_graph_with_interactions.png", community_membership)
        
        # Step 5: Calculate bonus factors - only for valid users
        all_users = set(gaming_scores.keys()) | set(impression_scores.keys()) | set(community_scores.keys()) | set(link_prediction_scores.keys())
        # Filter to only include users from impression data
        all_users = all_users.intersection(set(valid_impression_users))
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
        
        # Add aggregation and visualization
        self.perform_aggregation_and_visualization(df)
        
        return df
    
    def perform_aggregation_and_visualization(self, df: pd.DataFrame):
        """Perform data aggregation and visualization for level scores."""
        print("\nPerforming level score aggregation and visualization...")
        
        # Add timestamp column for aggregation (simulate different times)
        import numpy as np
        from datetime import datetime, timedelta
        
        # Simulate timestamps over the last 30 days
        start_date = datetime.now() - timedelta(days=30)
        timestamps = []
        for i in range(len(df)):
            # Simulate random timestamps over the last 30 days
            random_days = np.random.randint(0, 30)
            random_hours = np.random.randint(0, 24)
            timestamp = start_date + timedelta(days=random_days, hours=random_hours)
            timestamps.append(timestamp)
        
        df['timestamp'] = timestamps
        
        # Perform aggregations
        try:
            # Daily aggregation
            daily_agg = self.aggregation_utils.aggregate_level_data(
                df, period='daily', level_col='crew_level'
            )
            print(f"Daily aggregation: {len(daily_agg)} records")
            
            # Weekly aggregation
            weekly_agg = self.aggregation_utils.aggregate_level_data(
                df, period='weekly', level_col='crew_level'
            )
            print(f"Weekly aggregation: {len(weekly_agg)} records")
            
            # Monthly aggregation
            monthly_agg = self.aggregation_utils.aggregate_level_data(
                df, period='monthly', level_col='crew_level'
            )
            print(f"Monthly aggregation: {len(monthly_agg)} records")
            
        except Exception as e:
            print(f"Error during aggregation: {e}")
            daily_agg = weekly_agg = monthly_agg = pd.DataFrame()
        
        # Visualizations
        try:
            # 1. Level distribution
            self.visualization_utils.plot_level_distribution(
                df, level_col='crew_level',
                title="Crew Level Distribution",
                save_path="level_distribution.png"
            )
            
            # 2. Composite score distribution
            self.visualization_utils.plot_level_distribution(
                df, level_col='composite_score',
                title="Composite Score Distribution",
                save_path="composite_score_distribution.png"
            )
            
            # 3. Level trends for top users
            level_trends = self.aggregation_utils.get_level_trends(
                df, level_col='crew_level', top_n=10
            )
            
            if level_trends:
                self.visualization_utils.plot_level_trends(
                    level_trends,
                    title="Level Trends for Top 10 Users",
                    save_path="level_trends.png"
                )
            
            # 4. Aggregation visualizations
            if not daily_agg.empty:
                self.visualization_utils.plot_level_aggregations(
                    daily_agg, period='daily',
                    title="Daily Level Aggregations",
                    save_path="daily_level_aggregations.png"
                )
            
            if not weekly_agg.empty:
                self.visualization_utils.plot_level_aggregations(
                    weekly_agg, period='weekly',
                    title="Weekly Level Aggregations", 
                    save_path="weekly_level_aggregations.png"
                )
                
        except Exception as e:
            print(f"Error during visualization: {e}")
        
        print("Level aggregation and visualization completed!")


if __name__ == "__main__":
    calculator = LevelCalculator()
    results_df = calculator.calculate_crew_levels()
    
    if not results_df.empty:
        print(f"Successfully calculated levels for {len(results_df)} users")
    else:
        print("No results to save")
