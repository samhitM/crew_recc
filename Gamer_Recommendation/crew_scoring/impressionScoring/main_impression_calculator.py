"""
Main impression scoring calculator - modularized version.
"""
import os
import pandas as pd
from typing import Dict, List
import warnings

from database.db_manager import DatabaseManager
from graph.graph_manager import GraphManager
from scoring.scoring_manager import ScoringManager
from utils.helpers import NormalizationUtils, FileUtils

warnings.filterwarnings("ignore")


class ImpressionCalculator:
    """
    Modularized implementation for calculating crew impressions.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.graph_manager = GraphManager()
        self.scoring_manager = ScoringManager()
        self.normalizer = NormalizationUtils()
        self.file_utils = FileUtils()
    
    def calculate_final_impressions(self) -> pd.DataFrame:
        """Main method to calculate final impression scores with individual normalization."""
        print("Starting revised crew impression calculation with individual normalization...")
        
        # Check if results already exist
        output_file = "crew_impressions_revised.csv"
        if os.path.exists(output_file):
            print(f"Found existing impression results in {output_file}")
            try:
                existing_df = pd.read_csv(output_file)
                print(f"Loaded existing impression data for {len(existing_df)} users")
                return existing_df
            except Exception as e:
                print(f"Error reading existing file: {e}")
                print("Proceeding with fresh calculation...")
        
        # Step 1: Build graph and calculate metrics
        graph_metrics = self.graph_manager.calculate_graph_metrics()
        
        if not graph_metrics:
            print("No graph metrics available")
            return pd.DataFrame()
        
        # Visualize the graph after building
        self.graph_manager.visualize_graph("friendship_graph_with_interactions.png")
        
        # Step 2: Get raw data
        user_ids = list(graph_metrics.keys())
        message_data = self.db_manager.fetch_message_counts()
        interaction_data = self.db_manager.fetch_user_interactions()
        profile_likes_data = self.scoring_manager.calculate_profile_likes(interaction_data)
        
        # Step 3: Prepare feature data and learn weights
        feature_df, feature_names = self.scoring_manager.prepare_feature_data(graph_metrics)
        feature_weights = self.scoring_manager.learn_feature_weights(feature_df, graph_metrics)
        
        # Step 4: Calculate user feature scores
        user_feature_scores = self.scoring_manager.calculate_user_feature_scores(feature_df, feature_weights)
        
        # Step 5: Calculate website impressions
        website_impressions = self.scoring_manager.calculate_website_impressions(user_ids)
        
        # Step 6: Extract individual components for normalization
        raw_pagerank = {user: graph_metrics[user]['pagerank'] for user in user_ids}
        raw_k_shell = {user: graph_metrics[user]['k_shell'] for user in user_ids}
        raw_messages = {user: message_data.get(user, 0) for user in user_ids}
        raw_profile_likes = {user: profile_likes_data.get(user, 0) for user in user_ids}
        
        # Step 7: Normalize each component individually
        print("Normalizing each component individually...")
        norm_pagerank = self.normalizer.normalize_scores(raw_pagerank)
        norm_k_shell = self.normalizer.normalize_scores(raw_k_shell)
        norm_messages = self.normalizer.normalize_scores(raw_messages)
        norm_profile_likes = self.normalizer.normalize_scores(raw_profile_likes)
        norm_user_feature_scores = self.normalizer.normalize_scores(user_feature_scores)
        norm_website_impressions = self.normalizer.normalize_scores(website_impressions)
        
        # Step 8: Calculate topological score (pagerank + k_shell) and normalize
        raw_topological_scores = {}
        for user in user_ids:
            raw_topological_scores[user] = (
                self.scoring_manager.topological_weights['pagerank'] * norm_pagerank[user] +
                self.scoring_manager.topological_weights['k_shell'] * norm_k_shell[user]
            )
        norm_topological_scores = self.normalizer.normalize_scores(raw_topological_scores)
        
        # Step 9: Calculate total impression score and normalize
        raw_total_impression_scores = {}
        for user in user_ids:
            # Sum all normalized components for total impression
            raw_total_impression_scores[user] = (
                norm_topological_scores[user] +
                norm_user_feature_scores[user] +
                norm_website_impressions[user]
            )
        norm_total_impression_scores = self.normalizer.normalize_scores(raw_total_impression_scores)
        
        # Step 10: Create final dataframe with all normalized values
        results = []
        for user_id in user_ids:
            results.append({
                'user_id': user_id,
                'posts': 0,  # Keep posts as 0 since table is empty
                'messages': raw_messages[user_id],
                'profile_likes': raw_profile_likes[user_id],
                'pagerank': raw_pagerank[user_id],
                'k_shell': raw_k_shell[user_id],
                'user_feature_score': user_feature_scores.get(user_id, 0),
                'website_impressions': website_impressions.get(user_id, 0),
                'topological_score': raw_topological_scores[user_id],
                'total_impression_score': raw_total_impression_scores[user_id],
                # Normalized values
                'norm_messages': norm_messages[user_id],
                'norm_profile_likes': norm_profile_likes[user_id],
                'norm_pagerank': norm_pagerank[user_id],
                'norm_k_shell': norm_k_shell[user_id],
                'norm_user_feature_score': norm_user_feature_scores[user_id],
                'norm_website_impressions': norm_website_impressions[user_id],
                'norm_topological_score': norm_topological_scores[user_id],
                'norm_total_impression_score': norm_total_impression_scores[user_id]
            })
        
        df = pd.DataFrame(results)
        print(f"Calculated impressions for {len(df)} users")
        
        # Save results with error handling
        self.file_utils.save_results_with_fallback(df, output_file)
        
        return df


if __name__ == "__main__":
    calculator = ImpressionCalculator()
    results_df = calculator.calculate_final_impressions()
    
    if not results_df.empty:
        print(f"Successfully calculated impressions for {len(results_df)} users")
    else:
        print("No results to save")
