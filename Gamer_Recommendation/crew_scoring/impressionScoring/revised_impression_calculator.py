import sys
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings

# Add the main project directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from database.queries import fetch_all_users_data
from database.connection import get_db_connection, release_db_connection

warnings.filterwarnings("ignore")

class RevisedCrewImpressionCalculator:
    """
    Revised implementation for calculating crew impressions based on:
    1. Graph-based topological scores (PageRank, K-Shell, Out-Degree)
    2. Data-driven feature weights using linear regression
    3. Website impressions from profile visits
    """
    
    def __init__(self):
        self.graph = None
        self.pagerank_scores = {}
        self.k_shell_scores = {}
        self.out_degree_scores = {}
        self.feature_weights = {}
        self.scaler = StandardScaler()
        
        # Default weights for topological score
        self.topological_weights = {
            'pagerank': 0.4,
            'k_shell': 0.3, 
            'out_degree': 0.3
        }
        
        # Default feature weights (fallback if regression fails)
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
            'posts_on_topic': 0.05
        }
    
    def fetch_friendship_data(self) -> List[Dict]:
        """Fetch friendship relations from the database."""
        try:
            return fetch_all_users_data(
                table="friendship",
                database_name="crewdb", 
                columns=["user_a_id", "user_b_id", "relation"],
                conditions=[{"field": "state", "operator": "=", "value": True}]
            )
        except Exception as e:
            print(f"Error fetching friendship data: {e}")
            return []
    
    def fetch_user_games_data(self) -> Dict[str, float]:
        """Fetch gaming time data from user_games table."""
        try:
            records = fetch_all_users_data(
                table="user_games",
                database_name="crewdb",
                columns=["user_id", "gaming_time"]
            )
            return {record['user_id']: float(record.get('gaming_time', 0) or 0) for record in records}
        except Exception as e:
            print(f"Error fetching user games data: {e}")
            return {}
    
    def build_friendship_graph(self) -> nx.Graph:
        """Build a graph from friendship data."""
        print("Building friendship graph...")
        self.graph = nx.Graph()
        friendship_data = self.fetch_friendship_data()
        
        for record in friendship_data:
            user_a = record['user_a_id']
            user_b = record['user_b_id'] 
            relation_data = record.get('relation', {})
            
            if not relation_data:
                continue
                
            try:
                # Parse JSON if it's a string
                if isinstance(relation_data, str):
                    relation_data = json.loads(relation_data)
                
                # Add nodes
                self.graph.add_node(user_a)
                self.graph.add_node(user_b)
                
                # Check if they are friends based on the relation structure
                # Look for accepted status or follows: true
                is_friends = False
                for key, value in relation_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                status = sub_value.get('status', '')
                                follows = sub_value.get('follows', False)
                                if status == 'accepted' or follows:
                                    is_friends = True
                                    break
                        if is_friends:
                            break
                
                if is_friends:
                    self.graph.add_edge(user_a, user_b)
                    
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing relation data: {e}")
                continue
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def calculate_graph_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate PageRank, K-Shell, and Out-Degree for all users."""
        if self.graph is None:
            self.build_friendship_graph()
        
        print("Calculating graph metrics...")
        
        # Remove self-loops and calculate metrics
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        
        # PageRank
        self.pagerank_scores = nx.pagerank(self.graph, alpha=0.85)
        
        # K-Shell decomposition
        self.k_shell_scores = nx.core_number(self.graph)
        
        # Out-degree (number of connections)
        self.out_degree_scores = dict(self.graph.degree())
        
        # Combine all metrics
        all_users = set(self.pagerank_scores.keys()) | set(self.k_shell_scores.keys()) | set(self.out_degree_scores.keys())
        
        graph_metrics = {}
        for user in all_users:
            graph_metrics[user] = {
                'pagerank': self.pagerank_scores.get(user, 0.0),
                'k_shell': self.k_shell_scores.get(user, 0),
                'out_degree': self.out_degree_scores.get(user, 0)
            }
        
        return graph_metrics
    
    def calculate_topological_score(self, graph_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate topological score combining PageRank, K-Shell, and Out-Degree."""
        print("Calculating topological scores...")
        
        topological_scores = {}
        
        # Extract values for normalization
        pagerank_values = [metrics['pagerank'] for metrics in graph_metrics.values()]
        k_shell_values = [metrics['k_shell'] for metrics in graph_metrics.values()]
        out_degree_values = [metrics['out_degree'] for metrics in graph_metrics.values()]
        
        # Normalize to [0, 1] range
        def normalize_values(values):
            if not values or max(values) == min(values):
                return [0.0] * len(values)
            return [(v - min(values)) / (max(values) - min(values)) for v in values]
        
        norm_pagerank = normalize_values(pagerank_values)
        norm_k_shell = normalize_values(k_shell_values)
        norm_out_degree = normalize_values(out_degree_values)
        
        # Calculate weighted topological score
        for i, user in enumerate(graph_metrics.keys()):
            topological_score = (
                self.topological_weights['pagerank'] * norm_pagerank[i] +
                self.topological_weights['k_shell'] * norm_k_shell[i] +
                self.topological_weights['out_degree'] * norm_out_degree[i]
            )
            topological_scores[user] = topological_score
        
        return topological_scores
    
    def prepare_feature_data(self, graph_metrics: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature data for linear regression."""
        print("Preparing feature data...")
        
        # Get gaming time data
        gaming_data = self.fetch_user_games_data()
        
        # Create feature matrix
        features = []
        pagerank_targets = []
        user_ids = []
        
        for user_id, metrics in graph_metrics.items():
            # Default feature values (simulated since we don't have real data)
            feature_vector = {
                'reposts': 0,  # Default to 0 as specified
                'replies': 0,
                'mentions': 0,
                'favorites': 0,
                'interest_topic': 0,
                'bio_content': 0,
                'profile_likes': 0,
                'user_games': gaming_data.get(user_id, 0),  # Use actual gaming time
                'verified_status': 0,
                'posts_on_topic': 0
            }
            
            features.append(list(feature_vector.values()))
            pagerank_targets.append(metrics['pagerank'])
            user_ids.append(user_id)
        
        feature_names = list(feature_vector.keys())
        return pd.DataFrame(features, columns=feature_names, index=user_ids), feature_names
    
    def learn_feature_weights(self, feature_df: pd.DataFrame, graph_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Learn feature weights using linear regression with PageRank as target."""
        print("Learning feature weights using linear regression...")
        
        try:
            # Prepare target (PageRank scores)
            pagerank_targets = [graph_metrics[user]['pagerank'] for user in feature_df.index]
            
            # Check if we have variation in features and targets
            if len(set(pagerank_targets)) == 1:
                print("No variation in PageRank scores, using default weights")
                return self.default_feature_weights
            
            # Fit linear regression
            X = feature_df.values
            y = np.array(pagerank_targets)
            
            # Only fit if we have variation in features
            if np.var(X) == 0:
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
        
        # Default to 0 as specified, but can be enhanced with real data
        website_impressions = {}
        
        for user_id in user_ids:
            # Default formula: unique_pageviews * (1 + scroll_depth_percent/100)
            unique_pageviews = 0  # Default to 0
            scroll_depth_percent = 0  # Default to 0
            
            impression_score = unique_pageviews * (1 + scroll_depth_percent / 100)
            website_impressions[user_id] = impression_score
        
        return website_impressions
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to have mean 0 and std 1."""
        values = list(scores.values())
        if not values or np.std(values) == 0:
            return {k: 0.0 for k in scores.keys()}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        return {k: (v - mean_val) / std_val for k, v in scores.items()}
    
    def rescale_scores(self, normalized_scores: Dict[str, float], target_mean: float = 50, target_std: float = 15) -> Dict[str, float]:
        """Rescale normalized scores to a target range."""
        return {k: round(v * target_std + target_mean) for k, v in normalized_scores.items()}
    
    def calculate_final_impressions(self) -> pd.DataFrame:
        """Main method to calculate final impression scores."""
        print("Starting revised crew impression calculation...")
        
        # Step 1: Build graph and calculate metrics
        graph_metrics = self.calculate_graph_metrics()
        
        if not graph_metrics:
            print("No graph metrics available")
            return pd.DataFrame()
        
        # Step 2: Calculate topological scores
        topological_scores = self.calculate_topological_score(graph_metrics)
        
        # Step 3: Prepare feature data and learn weights
        feature_df, feature_names = self.prepare_feature_data(graph_metrics)
        feature_weights = self.learn_feature_weights(feature_df, graph_metrics)
        self.feature_weights = feature_weights
        
        # Step 4: Calculate user feature scores
        user_feature_scores = self.calculate_user_feature_scores(feature_df, feature_weights)
        
        # Step 5: Calculate website impressions
        user_ids = list(graph_metrics.keys())
        website_impressions = self.calculate_website_impressions(user_ids)
        
        # Step 6: Normalize all scores
        norm_topological = self.normalize_scores(topological_scores)
        norm_feature = self.normalize_scores(user_feature_scores)
        norm_impressions = self.normalize_scores(website_impressions)
        
        # Step 7: Rescale to meaningful ranges
        rescaled_topological = self.rescale_scores(norm_topological, 50, 15)
        rescaled_feature = self.rescale_scores(norm_feature, 30, 10)
        rescaled_impressions = self.rescale_scores(norm_impressions, 20, 8)
        
        # Step 8: Create final dataframe
        results = []
        for user_id in user_ids:
            pagerank = graph_metrics[user_id]['pagerank']
            total_impression = (
                rescaled_topological.get(user_id, 0) + 
                rescaled_feature.get(user_id, 0) + 
                rescaled_impressions.get(user_id, 0)
            )
            
            results.append({
                'user_id': user_id,
                'posts': 0,  # Default to 0 as specified
                'messages': 0,  # Default to 0 as specified
                'pagerank': pagerank,
                'k_shell': graph_metrics[user_id]['k_shell'],
                'out_degree': graph_metrics[user_id]['out_degree'],
                'topological_score': rescaled_topological.get(user_id, 0),
                'user_feature_score': rescaled_feature.get(user_id, 0),
                'website_impressions': rescaled_impressions.get(user_id, 0),
                'total_impression_score': total_impression
            })
        
        df = pd.DataFrame(results)
        print(f"Calculated impressions for {len(df)} users")
        return df
    
    def plot_impressions(self, df: pd.DataFrame, output_path: str = "impression_plots.png"):
        """Plot impressions vs user_id."""
        if df.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot total impression scores
        plt.subplot(2, 2, 1)
        plt.scatter(range(len(df)), df['total_impression_score'], alpha=0.6)
        plt.title('Total Impression Score vs User Index')
        plt.xlabel('User Index')
        plt.ylabel('Total Impression Score')
        
        # Plot PageRank scores
        plt.subplot(2, 2, 2)
        plt.scatter(range(len(df)), df['pagerank'], alpha=0.6, color='orange')
        plt.title('PageRank vs User Index')
        plt.xlabel('User Index')
        plt.ylabel('PageRank')
        
        # Plot score distribution
        plt.subplot(2, 2, 3)
        plt.hist(df['total_impression_score'], bins=20, alpha=0.7)
        plt.title('Distribution of Total Impression Scores')
        plt.xlabel('Total Impression Score')
        plt.ylabel('Frequency')
        
        # Plot component scores
        plt.subplot(2, 2, 4)
        plt.scatter(range(len(df)), df['topological_score'], alpha=0.6, label='Topological', s=20)
        plt.scatter(range(len(df)), df['user_feature_score'], alpha=0.6, label='Feature', s=20)
        plt.scatter(range(len(df)), df['website_impressions'], alpha=0.6, label='Web', s=20)
        plt.title('Component Scores vs User Index')
        plt.xlabel('User Index')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    calculator = RevisedCrewImpressionCalculator()
    results_df = calculator.calculate_final_impressions()
    
    if not results_df.empty:
        # Save results
        output_file = "crew_impressions_revised.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Plot results
        calculator.plot_impressions(results_df)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(results_df[['pagerank', 'topological_score', 'user_feature_score', 'total_impression_score']].describe())
    else:
        print("No results to save")
