import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings
import psycopg2
import json

warnings.filterwarnings("ignore")

class StandaloneCrewLevelCalculator:
    """
    Standalone revised implementation for calculating crew levels based on:
    1. Gaming Activity Score (using gaming_time from user_games table)
    2. Impression Score (from impression scoring)
    3. Community Detection Score (Louvain algorithm)
    4. Bonus Factors
    """
    
    def __init__(self):
        self.graph = None
        self.communities = {}
        self.impression_scores = {}
        
        # Database configuration
        self.db_config = {
            "host": "34.44.52.84",
            "port": 5432,
            "user": "admin_crew",
            "password": "xV/nI2+=uOI&KL1P",
            "database": "crewdb"
        }
        
        # Weights for final composite score
        self.composite_weights = {
            'gaming': 0.30,
            'impression': 0.25,
            'community': 0.10,
            'link_prediction': 0.20,  # Currently not implemented
            'bonus': 0.15
        }
        
        # Gaming activity weights
        self.gaming_weights = {
            'max_hours': 0.5,
            'achievements': 0.3,
            'special_achievements': 0.2
        }
        
        # Bonus factor weights
        self.bonus_weights = {
            'consistent_engagement': 0.30,
            'team_participation': 0.25,
            'community_contributions': 0.20,
            'verified_status': 0.15,
            'event_participation': 0.10
        }
        
        # Level thresholds (will be calculated dynamically)
        self.level_thresholds = []
        
        # Threshold calculation method ('percentile' or 'knn')
        self.threshold_method = 'percentile'
        
        # KNN parameters
        self.knn_neighbors = 5
        self.scaler = StandardScaler()
    
    def get_db_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def fetch_user_games_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch gaming data from user_games table."""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cur:
                query = "SELECT user_id, gaming_time FROM user_games LIMIT 10000"
                cur.execute(query)
                results = cur.fetchall()
                
                gaming_data = {}
                for row in results:
                    user_id = row[0]
                    gaming_time = float(row[1] or 0)
                    
                    # Convert gaming time to gaming activity components
                    # Since we only have gaming_time, we'll derive other metrics from it
                    gaming_data[user_id] = {
                        'max_hours': gaming_time,
                        'achievements': min(50, gaming_time / 10),  # Derived: 1 achievement per 10 hours
                        'special_achievements': min(10, gaming_time / 50)  # Derived: 1 special per 50 hours
                    }
                
                print(f"Fetched gaming data for {len(gaming_data)} users")
                return gaming_data
                
        except Exception as e:
            print(f"Error fetching user games data: {e}")
            return {}
        finally:
            conn.close()
    
    def fetch_friendship_data(self) -> List[Dict]:
        """Fetch friendship relations from the database."""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cur:
                query = """
                SELECT user_a_id, user_b_id, relation 
                FROM friendship 
                WHERE state = true
                LIMIT 10000
                """
                cur.execute(query)
                results = cur.fetchall()
                
                friendship_data = []
                for row in results:
                    friendship_data.append({
                        'user_a_id': row[0],
                        'user_b_id': row[1],
                        'relation': row[2]
                    })
                
                print(f"Fetched {len(friendship_data)} friendship records for community detection")
                return friendship_data
                
        except Exception as e:
            print(f"Error fetching friendship data: {e}")
            return []
        finally:
            conn.close()
    
    def build_community_graph(self) -> nx.Graph:
        """Build a graph for community detection."""
        print("Building community graph...")
        self.graph = nx.Graph()
        friendship_data = self.fetch_friendship_data()
        
        edges_added = 0
        
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
                
                # Check if they are friends
                is_friends = False
                if isinstance(relation_data, dict):
                    for key, value in relation_data.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    status = sub_value.get('status', '')
                                    follows = sub_value.get('follows', False)
                                    if status in ['accepted', 'friends'] or follows:
                                        is_friends = True
                                        break
                            if is_friends:
                                break
                
                if is_friends:
                    # Add edge with weight (co-play frequency or message count)
                    # For now, use weight = 1, can be enhanced with actual interaction data
                    self.graph.add_edge(user_a, user_b, weight=1)
                    edges_added += 1
                    
            except (json.JSONDecodeError, TypeError) as e:
                continue
        
        print(f"Community graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def calculate_gaming_activity_score(self, gaming_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate gaming activity scores for all users."""
        print("Calculating gaming activity scores...")
        
        gaming_scores = {}
        
        for user_id, data in gaming_data.items():
            score = (
                data['max_hours'] * self.gaming_weights['max_hours'] +
                data['achievements'] * self.gaming_weights['achievements'] +
                data['special_achievements'] * self.gaming_weights['special_achievements']
            )
            gaming_scores[user_id] = score
        
        # Normalize scores to 0-1 range
        if gaming_scores:
            max_score = max(gaming_scores.values())
            if max_score > 0:
                gaming_scores = {k: v / max_score for k, v in gaming_scores.items()}
        
        return gaming_scores
    
    def get_impression_scores(self) -> Dict[str, float]:
        """Get impression scores from the CSV file created by impression calculator."""
        print("Getting impression scores...")
        
        try:
            # Look for the impression scores file in the impressionScoring folder
            impression_file = "../impressionScoring/crew_impressions_revised.csv"
            if os.path.exists(impression_file):
                df = pd.read_csv(impression_file)
                
                # Use raw impression scores (not normalized) for better level differentiation
                impression_scores = {}
                for _, row in df.iterrows():
                    impression_scores[row['user_id']] = row['total_impression_score']
                
                print(f"Loaded impression scores for {len(impression_scores)} users from {impression_file}")
                return impression_scores
            else:
                # Try alternative locations
                alt_paths = [
                    "crew_impressions_revised.csv",  # Current directory
                    "../../crew_impressions_revised.csv",  # Parent directory
                    "../crew_impressions_revised.csv"  # One level up
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        df = pd.read_csv(alt_path)
                        impression_scores = {}
                        for _, row in df.iterrows():
                            impression_scores[row['user_id']] = row['total_impression_score']
                        print(f"Loaded impression scores for {len(impression_scores)} users from {alt_path}")
                        return impression_scores
                
                print("No impression scores file found in any expected location, using default values")
                return {}
                
        except Exception as e:
            print(f"Error getting impression scores: {e}")
            return {}
    
    def calculate_link_prediction_scores(self) -> Dict[str, float]:
        """Calculate link prediction scores using Jaccard and Katz centrality."""
        print("Calculating link prediction scores...")
        
        if self.graph is None:
            self.build_community_graph()
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph, cannot calculate link prediction")
            return {}
        
        try:
            link_scores = {}
            
            # 1. Jaccard coefficient-based prediction
            jaccard_scores = {}
            nodes = list(self.graph.nodes())
            
            for node in nodes:
                jaccard_sum = 0
                neighbor_count = 0
                
                # Calculate average Jaccard coefficient with potential links
                for other_node in nodes:
                    if node != other_node and not self.graph.has_edge(node, other_node):
                        # Calculate Jaccard coefficient
                        neighbors_node = set(self.graph.neighbors(node))
                        neighbors_other = set(self.graph.neighbors(other_node))
                        
                        if len(neighbors_node) > 0 or len(neighbors_other) > 0:
                            intersection = len(neighbors_node.intersection(neighbors_other))
                            union = len(neighbors_node.union(neighbors_other))
                            jaccard = intersection / union if union > 0 else 0
                            jaccard_sum += jaccard
                            neighbor_count += 1
                
                jaccard_scores[node] = jaccard_sum / neighbor_count if neighbor_count > 0 else 0
            
            # 2. Katz centrality-based prediction
            try:
                # Calculate Katz centrality with small alpha to avoid convergence issues
                alpha = 1.0 / (max(dict(self.graph.degree()).values()) + 1) if self.graph.number_of_edges() > 0 else 0.1
                katz_scores = nx.katz_centrality(self.graph, alpha=alpha, max_iter=1000, tol=1e-6)
            except:
                # Fallback to degree centrality if Katz fails
                print("Katz centrality failed, using degree centrality as fallback")
                degree_scores = dict(self.graph.degree())
                max_degree = max(degree_scores.values()) if degree_scores else 1
                katz_scores = {k: v / max_degree for k, v in degree_scores.items()}
            
            # 3. Combine Jaccard and Katz scores
            for node in nodes:
                jaccard_score = jaccard_scores.get(node, 0)
                katz_score = katz_scores.get(node, 0)
                
                # Weighted combination: 60% Jaccard, 40% Katz
                combined_score = 0.6 * jaccard_score + 0.4 * katz_score
                link_scores[node] = combined_score
            
            # Normalize to 0-1 range
            if link_scores:
                max_score = max(link_scores.values())
                if max_score > 0:
                    link_scores = {k: v / max_score for k, v in link_scores.items()}
            
            print(f"Calculated link prediction scores for {len(link_scores)} users")
            return link_scores
            
        except Exception as e:
            print(f"Error calculating link prediction scores: {e}")
            return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using connected components and link prediction enhancement."""
        print("Detecting communities with link prediction...")
        
        if self.graph is None:
            self.build_community_graph()
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph, cannot detect communities")
            return {}
        
        try:
            # Get link prediction scores for community enhancement
            link_scores = self.calculate_link_prediction_scores()
            
            # Use connected components as base communities
            communities = {}
            community_id = 0
            
            for component in nx.connected_components(self.graph):
                for node in component:
                    communities[node] = community_id
                community_id += 1
            
            # Enhance communities using link prediction
            # Users with high link prediction scores get community bonuses
            enhanced_communities = communities.copy()
            
            # Calculate community quality scores based on link prediction
            community_link_scores = {}
            for user_id, comm_id in communities.items():
                if comm_id not in community_link_scores:
                    community_link_scores[comm_id] = []
                community_link_scores[comm_id].append(link_scores.get(user_id, 0))
            
            # Store community quality for later use
            self.community_quality = {}
            for comm_id, scores in community_link_scores.items():
                self.community_quality[comm_id] = np.mean(scores) if scores else 0
            
            num_communities = len(set(enhanced_communities.values()))
            print(f"Detected {num_communities} communities with link prediction enhancement")
            
            return enhanced_communities
        except Exception as e:
            print(f"Error in community detection: {e}")
            return {}
    
    def calculate_community_scores(self) -> Dict[str, float]:
        """Calculate community detection scores enhanced with link prediction."""
        print("Calculating community scores with link prediction...")
        
        communities = self.detect_communities()
        if not communities:
            return {}
        
        try:
            # Get link prediction scores
            link_scores = self.calculate_link_prediction_scores()
            
            # Group users by community
            community_groups = {}
            for user_id, community_id in communities.items():
                if community_id not in community_groups:
                    community_groups[community_id] = []
                community_groups[community_id].append(user_id)
            
            # Calculate enhanced community scores
            community_scores = {}
            total_users = len(communities)
            
            for community_id, members in community_groups.items():
                community_size = len(members)
                
                # Base community score
                base_score = community_size / total_users
                
                # Community quality bonus from link prediction
                quality_bonus = getattr(self, 'community_quality', {}).get(community_id, 0) * 0.3
                
                # Calculate individual scores for members
                for user_id in members:
                    # Individual link prediction score
                    link_score = link_scores.get(user_id, 0)
                    
                    # Impression score influence
                    impression_score = self.impression_scores.get(user_id, 0)
                    
                    # Combined community score
                    final_score = (
                        base_score * 0.4 +                    # Community size influence
                        quality_bonus * 0.3 +                 # Community quality
                        link_score * 0.2 +                    # Individual link prediction
                        impression_score * 0.1                # Impression influence
                    )
                    
                    community_scores[user_id] = final_score
            
            # Normalize community scores to 0-1 range
            if community_scores:
                max_score = max(community_scores.values())
                if max_score > 0:
                    community_scores = {k: v / max_score for k, v in community_scores.items()}
            
            print(f"Calculated enhanced community scores for {len(community_scores)} users")
            return community_scores
            
        except Exception as e:
            print(f"Error calculating community scores: {e}")
            return {}
    
    def calculate_bonus_factors(self, user_ids: List[str]) -> Dict[str, float]:
        """Calculate bonus factors scores."""
        print("Calculating bonus factors...")
        
        bonus_scores = {}
        
        for user_id in user_ids:
            # Default values for bonus factors (can be enhanced with real data)
            factors = {
                'consistent_engagement': 0.5,  # Normalized login streaks
                'team_participation': 0.4,     # Normalized team events
                'community_contributions': 0.4, # Normalized forum posts
                'verified_status': 0.0,        # 1 if verified, 0 otherwise
                'event_participation': 0.5     # Normalized special events
            }
            
            # Calculate weighted bonus score
            bonus_score = sum(
                factors[factor] * weight 
                for factor, weight in self.bonus_weights.items()
            )
            
            bonus_scores[user_id] = bonus_score
        
        return bonus_scores
    
    def calculate_composite_scores(self, gaming_scores: Dict[str, float], 
                                 impression_scores: Dict[str, float],
                                 community_scores: Dict[str, float],
                                 bonus_scores: Dict[str, float],
                                 link_prediction_scores: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate final composite scores with link prediction."""
        print("Calculating composite scores with link prediction...")
        
        # Get link prediction scores if not provided
        if link_prediction_scores is None:
            link_prediction_scores = self.calculate_link_prediction_scores()
        
        all_users = set(gaming_scores.keys()) | set(impression_scores.keys()) | set(community_scores.keys()) | set(bonus_scores.keys()) | set(link_prediction_scores.keys())
        
        # Normalize impression scores to 0-1 range for composite calculation
        normalized_impression_scores = {}
        if impression_scores:
            max_impression = max(impression_scores.values()) if impression_scores.values() else 1
            if max_impression > 0:
                normalized_impression_scores = {k: v / max_impression for k, v in impression_scores.items()}
            else:
                normalized_impression_scores = {k: 0 for k in impression_scores.keys()}
        
        composite_scores = {}
        
        for user_id in all_users:
            gaming = gaming_scores.get(user_id, 0)
            impression = normalized_impression_scores.get(user_id, 0)  # Use normalized for composite
            community = community_scores.get(user_id, 0)
            bonus = bonus_scores.get(user_id, 0)
            link_prediction = link_prediction_scores.get(user_id, 0)
            
            composite_score = (
                self.composite_weights['gaming'] * gaming +
                self.composite_weights['impression'] * impression +
                self.composite_weights['community'] * community +
                self.composite_weights['link_prediction'] * link_prediction +
                self.composite_weights['bonus'] * bonus
            )
            
            composite_scores[user_id] = composite_score
        
        print(f"Calculated composite scores for {len(composite_scores)} users")
        return composite_scores
    
    def calculate_level_thresholds(self, composite_scores: Dict[str, float], num_levels: int = 5, method: str = 'percentile') -> List[float]:
        """Calculate level thresholds based on score distribution."""
        print(f"Calculating level thresholds for {num_levels} levels using {method} method...")
        
        if not composite_scores:
            return []
        
        if method == 'percentile':
            scores = list(composite_scores.values())
            scores.sort()
            
            # Calculate percentile-based thresholds
            thresholds = []
            for i in range(1, num_levels):
                percentile = i * (100 / num_levels)
                threshold = np.percentile(scores, percentile)
                thresholds.append(threshold)
            
            print(f"Percentile-based thresholds: {thresholds}")
            return thresholds
        
        elif method == 'knn':
            # Enhanced KNN-based thresholding with clustering analysis
            return self._calculate_knn_thresholds(composite_scores, num_levels)
        
        else:
            print("Invalid method specified for threshold calculation")
            return []
    
    def assign_levels(self, composite_scores: Dict[str, float], thresholds: List[float]) -> Dict[str, int]:
        """Assign crew levels based on composite scores and thresholds."""
        print("Assigning crew levels...")
        
        level_assignments = {}
        
        for user_id, score in composite_scores.items():
            level = 1  # Start with level 1
            
            for threshold in thresholds:
                if score >= threshold:
                    level += 1
                else:
                    break
            
            level_assignments[user_id] = level
        
        # Print level distribution
        level_counts = {}
        for level in level_assignments.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("Level distribution:")
        for level, count in sorted(level_counts.items()):
            print(f"  Level {level}: {count} users")
        
        return level_assignments
    
    def calculate_crew_levels(self, threshold_method: str = None) -> pd.DataFrame:
        """Main method to calculate crew levels.
        
        Args:
            threshold_method: 'percentile' or 'knn'. If None, uses self.threshold_method
        """
        print("Starting revised crew level calculation...")
        
        if threshold_method is None:
            threshold_method = self.threshold_method
        
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
        gaming_data = self.fetch_user_games_data()
        gaming_scores = self.calculate_gaming_activity_score(gaming_data)
        
        # Step 2: Get impression scores
        self.impression_scores = self.get_impression_scores()
        
        # Step 3: Calculate community scores
        community_scores = self.calculate_community_scores()
        
        # Step 4: Calculate link prediction scores
        link_prediction_scores = self.calculate_link_prediction_scores()
        
        # Step 5: Calculate bonus factors
        all_users = set(gaming_scores.keys()) | set(self.impression_scores.keys()) | set(community_scores.keys()) | set(link_prediction_scores.keys())
        bonus_scores = self.calculate_bonus_factors(list(all_users))
        
        # Step 6: Calculate composite scores
        composite_scores = self.calculate_composite_scores(
            gaming_scores, self.impression_scores, community_scores, bonus_scores, link_prediction_scores
        )
        
        # Step 7: Calculate thresholds and assign levels
        thresholds = self.calculate_level_thresholds(composite_scores, method=threshold_method)
        level_assignments = self.assign_levels(composite_scores, thresholds)
        
        # Step 8: Create results dataframe
        results = []
        for user_id in all_users:
            results.append({
                'user_id': user_id,
                'gaming_score': gaming_scores.get(user_id, 0),
                'impression_score': self.impression_scores.get(user_id, 0),
                'community_score': community_scores.get(user_id, 0),
                'link_prediction_score': link_prediction_scores.get(user_id, 0),
                'bonus_score': bonus_scores.get(user_id, 0),
                'composite_score': composite_scores.get(user_id, 0),
                'crew_level': level_assignments.get(user_id, 1),
                'gaming_time': gaming_data.get(user_id, {}).get('max_hours', 0)
            })
        
        df = pd.DataFrame(results)
        print(f"Calculated crew levels for {len(df)} users")
        
        # Save results with error handling
        try:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        except PermissionError:
            print(f"Permission denied saving to {output_file}. File may be open in another program.")
            # Try alternative filename
            alt_file = f"crew_levels_revised_{int(pd.Timestamp.now().timestamp())}.csv"
            df.to_csv(alt_file, index=False)
            print(f"Results saved to alternative file: {alt_file}")
        
        return df
    
    def plot_level_distribution(self, df: pd.DataFrame, output_path: str = "level_plots.png"):
        """Plot level distribution and score analysis."""
        if df.empty:
            print("No data to plot")
            return
        
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Level distribution
            plt.subplot(2, 3, 1)
            level_counts = df['crew_level'].value_counts().sort_index()
            plt.bar(level_counts.index, level_counts.values)
            plt.title('Crew Level Distribution')
            plt.xlabel('Crew Level')
            plt.ylabel('Number of Users')
            
            # Plot 2: Composite score distribution
            plt.subplot(2, 3, 2)
            plt.hist(df['composite_score'], bins=20, alpha=0.7)
            plt.title('Composite Score Distribution')
            plt.xlabel('Composite Score')
            plt.ylabel('Frequency')
            
            # Plot 3: Gaming time vs Level
            plt.subplot(2, 3, 3)
            for level in sorted(df['crew_level'].unique()):
                level_data = df[df['crew_level'] == level]
                plt.scatter(level_data['gaming_time'], [level] * len(level_data), 
                           alpha=0.6, label=f'Level {level}')
            plt.title('Gaming Time vs Crew Level')
            plt.xlabel('Gaming Time (hours)')
            plt.ylabel('Crew Level')
            plt.legend()
            
            # Plot 4: Score components by level
            plt.subplot(2, 3, 4)
            score_cols = ['gaming_score', 'impression_score', 'community_score', 'bonus_score']
            level_means = df.groupby('crew_level')[score_cols].mean()
            
            x = np.arange(len(level_means.index))
            width = 0.2
            
            for i, col in enumerate(score_cols):
                plt.bar(x + i*width, level_means[col], width, label=col.replace('_', ' ').title())
            
            plt.title('Average Score Components by Level')
            plt.xlabel('Crew Level')
            plt.ylabel('Average Score')
            plt.xticks(x + width*1.5, level_means.index)
            plt.legend()
            
            # Plot 5: Composite score vs Level
            plt.subplot(2, 3, 5)
            plt.boxplot([df[df['crew_level'] == level]['composite_score'] 
                        for level in sorted(df['crew_level'].unique())],
                       labels=sorted(df['crew_level'].unique()))
            plt.title('Composite Score Distribution by Level')
            plt.xlabel('Crew Level')
            plt.ylabel('Composite Score')
            
            # Plot 6: Gaming time distribution
            plt.subplot(2, 3, 6)
            plt.hist(df['gaming_time'], bins=20, alpha=0.7)
            plt.title('Gaming Time Distribution')
            plt.xlabel('Gaming Time (hours)')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Level plots saved to {output_path}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def _calculate_knn_thresholds(self, composite_scores: Dict[str, float], 
                                 num_levels: int) -> List[float]:
        """Calculate KNN-based thresholds using clustering."""
        scores = list(composite_scores.values())
        
        if len(scores) < 4:
            print("Too few scores for KNN clustering, using percentile method")
            return self._calculate_percentile_fallback(scores, num_levels)
        
        try:
            # Find optimal number of clusters if not specified
            optimal_clusters = self._find_optimal_clusters(scores, num_levels)
            
            # Perform K-means clustering on scores
            scores_array = np.array(scores).reshape(-1, 1)
            scores_normalized = self.scaler.fit_transform(scores_array)
            
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scores_normalized)
            
            # Create feature matrix for KNN
            features = []
            for i, score in enumerate(scores):
                features.append([
                    score,  # Original composite score
                    i / len(scores),  # Rank position (normalized)
                    scores_normalized[i][0],  # Normalized score
                ])
            
            features_array = np.array(features)
            
            # Train KNN classifier on cluster assignments
            knn = KNeighborsClassifier(n_neighbors=min(self.knn_neighbors, len(scores) // 2))
            knn.fit(features_array, cluster_labels)
            
            # Calculate cluster centroids and boundaries
            cluster_info = {}
            for level in range(optimal_clusters):
                cluster_mask = cluster_labels == level
                cluster_scores = np.array(scores)[cluster_mask]
                
                if len(cluster_scores) > 0:
                    cluster_info[level] = {
                        'min_score': cluster_scores.min(),
                        'max_score': cluster_scores.max(),
                        'mean_score': cluster_scores.mean(),
                        'size': len(cluster_scores)
                    }
            
            # Calculate thresholds as boundaries between clusters
            # Sort clusters by mean score
            sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['mean_score'])
            
            thresholds = []
            for i in range(len(sorted_clusters) - 1):
                current_cluster = sorted_clusters[i][1]
                next_cluster = sorted_clusters[i + 1][1]
                
                # Threshold is midpoint between cluster boundaries
                threshold = (current_cluster['max_score'] + next_cluster['min_score']) / 2
                thresholds.append(threshold)
            
            print(f"KNN-based thresholds: {thresholds}")
            print("Cluster information:")
            for i, (cluster_id, info) in enumerate(sorted_clusters):
                print(f"  Level {i+1}: {info['size']} users, "
                      f"score range: {info['min_score']:.3f}-{info['max_score']:.3f}, "
                      f"mean: {info['mean_score']:.3f}")
            
            return thresholds
            
        except Exception as e:
            print(f"Error in KNN thresholding: {e}, falling back to percentile method")
            return self._calculate_percentile_fallback(scores, num_levels)
    
    def _find_optimal_clusters(self, scores: List[float], max_clusters: int = 8) -> int:
        """Find optimal number of clusters using silhouette score."""
        if len(scores) < 4:
            return min(len(scores), 3)
        
        scores_array = np.array(scores).reshape(-1, 1)
        scores_normalized = self.scaler.fit_transform(scores_array)
        
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(scores)))
        
        best_score = -1
        best_clusters = max_clusters
        
        for n_clusters in cluster_range:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scores_normalized)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(scores_normalized, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_clusters = n_clusters
                    
            except Exception as e:
                print(f"Error calculating silhouette for {n_clusters} clusters: {e}")
                continue
        
        if silhouette_scores:
            print(f"Silhouette scores: {dict(zip(cluster_range, silhouette_scores))}")
            print(f"Optimal clusters (silhouette): {best_clusters}")
            return best_clusters
        
        return min(max_clusters, len(scores) // 2) if len(scores) >= 4 else 2
    
    def _calculate_percentile_fallback(self, scores: List[float], num_levels: int) -> List[float]:
        """Fallback to percentile-based thresholds."""
        print("Using fallback percentile-based thresholds")
        scores = sorted(scores)
        thresholds = []
        
        for i in range(1, num_levels):
            percentile = i * (100 / num_levels)
            threshold = np.percentile(scores, percentile)
            thresholds.append(threshold)
        
        return thresholds

if __name__ == "__main__":
    calculator = StandaloneCrewLevelCalculator()
    results_df = calculator.calculate_crew_levels()
    
    if not results_df.empty:
        # Save results
        output_file = "crew_levels_revised.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Plot results
        calculator.plot_level_distribution(results_df)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(results_df[['gaming_score', 'impression_score', 'community_score', 'composite_score', 'crew_level']].describe())
        
        # Print level statistics
        print("\nLevel Statistics:")
        level_stats = results_df.groupby('crew_level').agg({
            'composite_score': ['mean', 'min', 'max'],
            'gaming_time': 'mean'
        }).round(3)
        print(level_stats)
    else:
        print("No results to save")
