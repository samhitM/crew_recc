import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from community import community_louvain
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from database.queries import fetch_all_users_data
from database.connection import get_db_connection, release_db_connection

warnings.filterwarnings("ignore")

class RevisedCrewLevelCalculator:
    """
    Revised implementation for calculating crew levels based on:
    1. Gaming Activity Score (using gaming_time from user_games table)
    2. Impression Score (from impression scoring)
    3. Community Detection Score (Louvain algorithm)
    4. Bonus Factors
    """
    
    def __init__(self):
        self.graph = None
        self.communities = {}
        self.impression_scores = {}
        
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
    
    def fetch_user_games_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch gaming data from user_games table."""
        try:
            records = fetch_all_users_data(
                table="user_games",
                database_name="crewdb",
                columns=["user_id", "gaming_time"]
            )
            
            gaming_data = {}
            for record in records:
                user_id = record['user_id']
                gaming_time = float(record.get('gaming_time', 0) or 0)
                
                # Convert gaming time to gaming activity components
                # Since we only have gaming_time, we'll derive other metrics from it
                gaming_data[user_id] = {
                    'max_hours': gaming_time,
                    'achievements': min(50, gaming_time / 10),  # Derived: 1 achievement per 10 hours
                    'special_achievements': min(10, gaming_time / 50)  # Derived: 1 special per 50 hours
                }
            
            return gaming_data
        except Exception as e:
            print(f"Error fetching user games data: {e}")
            return {}
    
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
    
    def build_community_graph(self) -> nx.Graph:
        """Build a graph for community detection."""
        print("Building community graph...")
        self.graph = nx.Graph()
        friendship_data = self.fetch_friendship_data()
        
        for record in friendship_data:
            user_a = record['user_a_id']
            user_b = record['user_b_id']
            relation_data = record.get('relation', {})
            
            if not relation_data:
                continue
                
            try:
                import json
                # Parse JSON if it's a string
                if isinstance(relation_data, str):
                    relation_data = json.loads(relation_data)
                
                # Add nodes
                self.graph.add_node(user_a)
                self.graph.add_node(user_b)
                
                # Check if they are friends
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
                    # Add edge with weight (co-play frequency or message count)
                    # For now, use weight = 1, can be enhanced with actual interaction data
                    self.graph.add_edge(user_a, user_b, weight=1)
                    
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing relation data: {e}")
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
        """Get impression scores from the revised impression calculator."""
        print("Getting impression scores...")
        
        try:
            # Import and run the revised impression calculator
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "impressionScoring")))
            from revised_impression_calculator import RevisedCrewImpressionCalculator
            
            impression_calc = RevisedCrewImpressionCalculator()
            results_df = impression_calc.calculate_final_impressions()
            
            if not results_df.empty:
                # Normalize impression scores to 0-1 range
                impression_scores = {}
                max_score = results_df['total_impression_score'].max()
                if max_score > 0:
                    for _, row in results_df.iterrows():
                        impression_scores[row['user_id']] = row['total_impression_score'] / max_score
                
                return impression_scores
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting impression scores: {e}")
            return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain algorithm."""
        print("Detecting communities using Louvain algorithm...")
        
        if self.graph is None:
            self.build_community_graph()
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph, cannot detect communities")
            return {}
        
        try:
            # Use Louvain algorithm for community detection
            self.communities = community_louvain.best_partition(self.graph)
            
            num_communities = len(set(self.communities.values()))
            print(f"Detected {num_communities} communities")
            
            return self.communities
        except Exception as e:
            print(f"Error in community detection: {e}")
            return {}
    
    def calculate_community_scores(self) -> Dict[str, float]:
        """Calculate community detection scores."""
        print("Calculating community scores...")
        
        communities = self.detect_communities()
        if not communities:
            return {}
        
        # Calculate modularity
        try:
            modularity = community_louvain.modularity(communities, self.graph)
        except:
            modularity = 0.5  # Default modularity
        
        # Group users by community
        community_groups = {}
        for user_id, community_id in communities.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(user_id)
        
        # Calculate base scores and identify top performers
        community_scores = {}
        total_users = len(communities)
        
        for community_id, members in community_groups.items():
            community_size = len(members)
            base_score = (community_size / total_users) * modularity
            
            # Get impression scores for ranking (fallback to gaming scores if not available)
            member_scores = {}
            for user_id in members:
                # Use impression score if available, otherwise use default
                member_scores[user_id] = self.impression_scores.get(user_id, 0)
            
            # Sort members by their scores (descending)
            sorted_members = sorted(member_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Assign scores
            top_performers_count = min(5, len(members))  # Top 5 or all if less than 5
            bonus = base_score * 0.5
            
            for i, (user_id, _) in enumerate(sorted_members):
                if i < top_performers_count:
                    community_scores[user_id] = base_score + bonus
                else:
                    community_scores[user_id] = base_score
        
        # Normalize community scores to 0-1 range
        if community_scores:
            max_score = max(community_scores.values())
            if max_score > 0:
                community_scores = {k: v / max_score for k, v in community_scores.items()}
        
        return community_scores
    
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
                                 bonus_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate final composite scores."""
        print("Calculating composite scores...")
        
        all_users = set(gaming_scores.keys()) | set(impression_scores.keys()) | set(community_scores.keys()) | set(bonus_scores.keys())
        
        composite_scores = {}
        
        for user_id in all_users:
            gaming = gaming_scores.get(user_id, 0)
            impression = impression_scores.get(user_id, 0)
            community = community_scores.get(user_id, 0)
            bonus = bonus_scores.get(user_id, 0)
            
            # Link prediction score (placeholder - not implemented)
            link_prediction = 0.5  # Default value
            
            composite_score = (
                self.composite_weights['gaming'] * gaming +
                self.composite_weights['impression'] * impression +
                self.composite_weights['community'] * community +
                self.composite_weights['link_prediction'] * link_prediction +
                self.composite_weights['bonus'] * bonus
            )
            
            composite_scores[user_id] = composite_score
        
        return composite_scores
    
    def calculate_level_thresholds(self, composite_scores: Dict[str, float], num_levels: int = 5) -> List[float]:
        """Calculate level thresholds based on score distribution."""
        print(f"Calculating level thresholds for {num_levels} levels...")
        
        if not composite_scores:
            return []
        
        scores = list(composite_scores.values())
        scores.sort()
        
        # Calculate percentile-based thresholds
        thresholds = []
        for i in range(1, num_levels):
            percentile = i * (100 / num_levels)
            threshold = np.percentile(scores, percentile)
            thresholds.append(threshold)
        
        print(f"Level thresholds: {thresholds}")
        return thresholds
    
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
    
    def calculate_crew_levels(self) -> pd.DataFrame:
        """Main method to calculate crew levels."""
        print("Starting revised crew level calculation...")
        
        # Step 1: Get gaming data and calculate gaming scores
        gaming_data = self.fetch_user_games_data()
        gaming_scores = self.calculate_gaming_activity_score(gaming_data)
        
        # Step 2: Get impression scores
        self.impression_scores = self.get_impression_scores()
        
        # Step 3: Calculate community scores
        community_scores = self.calculate_community_scores()
        
        # Step 4: Calculate bonus factors
        all_users = set(gaming_scores.keys()) | set(self.impression_scores.keys()) | set(community_scores.keys())
        bonus_scores = self.calculate_bonus_factors(list(all_users))
        
        # Step 5: Calculate composite scores
        composite_scores = self.calculate_composite_scores(
            gaming_scores, self.impression_scores, community_scores, bonus_scores
        )
        
        # Step 6: Calculate thresholds and assign levels
        thresholds = self.calculate_level_thresholds(composite_scores)
        level_assignments = self.assign_levels(composite_scores, thresholds)
        
        # Step 7: Create results dataframe
        results = []
        for user_id in all_users:
            results.append({
                'user_id': user_id,
                'gaming_score': gaming_scores.get(user_id, 0),
                'impression_score': self.impression_scores.get(user_id, 0),
                'community_score': community_scores.get(user_id, 0),
                'bonus_score': bonus_scores.get(user_id, 0),
                'composite_score': composite_scores.get(user_id, 0),
                'crew_level': level_assignments.get(user_id, 1),
                'gaming_time': gaming_data.get(user_id, {}).get('max_hours', 0)
            })
        
        df = pd.DataFrame(results)
        print(f"Calculated crew levels for {len(df)} users")
        return df
    
    def plot_level_distribution(self, df: pd.DataFrame, output_path: str = "level_plots.png"):
        """Plot level distribution and score analysis."""
        if df.empty:
            print("No data to plot")
            return
        
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
        
        # Plot 6: Score correlation matrix
        plt.subplot(2, 3, 6)
        corr_data = df[['gaming_score', 'impression_score', 'community_score', 'bonus_score', 'composite_score']].corr()
        plt.imshow(corr_data, cmap='coolwarm', aspect='auto')
        plt.title('Score Correlation Matrix')
        plt.colorbar()
        plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45)
        plt.yticks(range(len(corr_data.columns)), corr_data.columns)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Level plots saved to {output_path}")

if __name__ == "__main__":
    calculator = RevisedCrewLevelCalculator()
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
