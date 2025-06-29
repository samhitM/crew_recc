"""
Refactored Crew Level Scoring - Now uses modular components.
This file has been refactored from 1149+ lines to use modular components.

The original monolithic code has been split into:
- database/level_db_manager.py - Database operations
- graph/level_graph_manager.py - Graph construction and metrics
- scoring/level_scoring_manager.py - Scoring calculations
- clustering/level_clustering_manager.py - KNN and hybrid clustering
- utils/level_helpers.py - Helper functions
- main_level_calculator.py - Main orchestrator
"""
import warnings
from main_level_calculator import LevelCalculator

warnings.filterwarnings("ignore")

class StandaloneCrewLevelCalculator:
    """
    Refactored wrapper class that uses modular components.
    Original 1149+ lines reduced to simple delegation.
    """
    
    def __init__(self):
        self.calculator = LevelCalculator()
    
    def calculate_final_levels(self):
        """Calculate final levels using modular components."""
        return self.calculator.calculate_final_levels()

if __name__ == "__main__":
    # Use the modular calculator
    calculator = StandaloneCrewLevelCalculator()
    results_df = calculator.calculate_final_levels()
    
    if not results_df.empty:
        print(f"Successfully calculated levels for {len(results_df)} users")
        print("Results saved to crew_levels_revised.csv")
    else:
        print("No results to save")
    """
    Implementation for calculating crew levels based on:
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
    
    def fetch_user_interactions(self) -> List[Dict]:
        """Fetch user interactions from the database."""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cur:
                query = """
                SELECT user_id, entity_id_primary, interaction_type, action 
                FROM user_interactions 
                LIMIT 50000
                """
                cur.execute(query)
                results = cur.fetchall()
                
                interaction_data = []
                for row in results:
                    interaction_data.append({
                        'user_id': row[0],
                        'entity_id_primary': row[1],
                        'interaction_type': row[2],
                        'action': row[3]
                    })
                
                print(f"Fetched {len(interaction_data)} user interaction records for community detection")
                return interaction_data
                
        except Exception as e:
            print(f"Error fetching user interactions data: {e}")
            return []
        finally:
            conn.close()
    
    def build_community_graph(self) -> nx.DiGraph:
        """Build a directed weighted graph for community detection and link prediction."""
        print("Building directed weighted community graph with user interactions...")
        self.graph = nx.DiGraph()  # Use directed graph
        friendship_data = self.fetch_friendship_data()
        interaction_data = self.fetch_user_interactions()
        
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
                
                # Extract relationship information and calculate edge weights
                if isinstance(relation_data, dict):
                    # Skip the outer key (it's not a user ID), go directly to user relationships
                    for outer_key, user_relations in relation_data.items():
                        if isinstance(user_relations, dict):
                            # Get all user IDs from this relation object
                            user_ids = list(user_relations.keys())
                            
                            # Create bidirectional edges between users based on their individual status
                            for i, user_a in enumerate(user_ids):
                                for j, user_b in enumerate(user_ids):
                                    if i != j:  # Don't create self-loops
                                        user_a_info = user_relations.get(user_a, {})
                                        if isinstance(user_a_info, dict):
                                            status = user_a_info.get('status', '').lower()
                                            follows = user_a_info.get('follows', False)
                                            
                                            # Calculate edge weight based on status and follows
                                            weight = self._calculate_edge_weight(status, follows)
                                            
                                            # Add directed edge from user_a to user_b
                                            if weight > 0:
                                                self.graph.add_edge(user_a, user_b, weight=weight)
                                                edges_added += 1
                    
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # Handle malformed JSON or unexpected structure
                continue
        
        # Add edges from user interactions
        print("Adding edges from user interactions...")
        interaction_edges_added = 0
        
        for interaction in interaction_data:
            user_id = interaction.get('user_id')
            entity_id = interaction.get('entity_id_primary')
            interaction_type = interaction.get('interaction_type', '').upper()
            action = interaction.get('action', '').lower()
            
            if user_id and entity_id and user_id != entity_id:
                # Add nodes if they don't exist
                self.graph.add_node(user_id)
                self.graph.add_node(entity_id)
                
                # Calculate interaction weight
                interaction_weight = self._calculate_interaction_weight(interaction_type, action)
                
                if interaction_weight > 0:
                    # Check if edge already exists and update weight
                    if self.graph.has_edge(user_id, entity_id):
                        current_weight = self.graph[user_id][entity_id]['weight']
                        # Combine weights (friendship + interaction)
                        new_weight = min(1.0, current_weight + interaction_weight * 0.3)  # Scale interaction weight
                        self.graph[user_id][entity_id]['weight'] = new_weight
                    else:
                        # Create new edge with interaction weight
                        self.graph.add_edge(user_id, entity_id, weight=interaction_weight)
                        interaction_edges_added += 1
        
        print(f"Added {interaction_edges_added} interaction edges")
        print(f"Directed weighted community graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _calculate_edge_weight(self, status: str, follows: bool) -> float:
        """Calculate edge weight based on relationship status and follow status."""
        # Base weight from status
        status_weights = {
            'friends': 1.0,
            'accepted': 1.0,
            'pending': 0.5,
            'request_sent': 0.3,
            'blocked': 0.1,
            'reported_list': 0.1,
            'declined': 0.1,
            'unknown': 0.1
        }
        
        # Get base weight from status
        base_weight = status_weights.get(status, 0.1)  # Default 0.1 for unknown statuses
        
        # Boost weight if user follows the other
        if follows:
            base_weight = min(1.0, base_weight + 0.4)  # Add 0.4 for follows, cap at 1.0
        
        return base_weight
    
    def _calculate_interaction_weight(self, interaction_type: str, action: str) -> float:
        """Calculate edge weight based on interaction type and action."""
        interaction_weights = {
            'PROFILE_INTERACTION': {
                'like': 0.6,
                'friend_request': 0.4,
                'ignored': 0.1
            },
            'SWIPE': {
                'like': 0.3,
                'friend_request': 0.2,
                'ignored': 0.05
            }
        }
        
        if interaction_type in interaction_weights:
            return interaction_weights[interaction_type].get(action, 0.1)
        else:
            return 0.1  # Default for unknown interaction types
    
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
        """Get normalized impression scores from the CSV file created by impression calculator."""
        print("Getting normalized impression scores...")
        
        try:
            # Look for the impression scores file in the impressionScoring folder
            impression_file = "../impressionScoring/crew_impressions_revised.csv"
            if os.path.exists(impression_file):
                df = pd.read_csv(impression_file)
                
                # Use normalized total impression scores for proper normalization in level scoring
                impression_scores = {}
                for _, row in df.iterrows():
                    impression_scores[row['user_id']] = row['norm_total_impression_score']
                
                print(f"Loaded normalized impression scores for {len(impression_scores)} users from {impression_file}")
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
                            impression_scores[row['user_id']] = row['norm_total_impression_score']
                        print(f"Loaded normalized impression scores for {len(impression_scores)} users from {alt_path}")
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
            # Visualize the community graph after building
            self.visualize_community_graph("community_graph_with_interactions.png")
        
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
        
        # Step 6: Calculate composite scores with normalization
        composite_scores, normalized_scores = self.calculate_composite_scores_normalized(
            gaming_scores, self.impression_scores, community_scores, bonus_scores, link_prediction_scores
        )
        
        # Step 7: Use KNN clustering to assign levels
        level_assignments = self.assign_levels_with_knn_clustering(composite_scores)
        
        # Step 8: Create results dataframe
        results = []
        for user_id in all_users:
            # Get normalized scores for this user
            norm_scores = normalized_scores.get(user_id, {})
            
            results.append({
                'user_id': user_id,
                'gaming_score': gaming_scores.get(user_id, 0),
                'impression_score': self.impression_scores.get(user_id, 0),
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
    
    def visualize_community_graph(self, filename: str = "community_graph.png"):
        """Visualize the community graph and save as PNG."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("No graph to visualize")
            return
        
        print(f"Visualizing community graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges...")
        
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = []
        node_colors = []
        for node in self.graph.nodes():
            # Size based on degree
            degree = self.graph.degree(node)
            node_sizes.append(max(50, degree * 20))
            
            # Color based on in-degree (influence)
            in_degree = self.graph.in_degree(node)
            node_colors.append(in_degree)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_size=node_sizes, 
                             node_color=node_colors,
                             cmap=plt.cm.plasma,
                             alpha=0.7)
        
        # Draw edges with weights
        edges = self.graph.edges()
        weights = [self.graph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(self.graph, pos,
                             width=[w * 2 for w in weights],
                             alpha=0.6,
                             edge_color=weights,
                             edge_cmap=plt.cm.Blues)
        
        # Add labels for high-degree nodes only (to avoid clutter)
        high_degree_nodes = {node: node for node in self.graph.nodes() 
                           if self.graph.degree(node) > 3}
        nx.draw_networkx_labels(self.graph, pos, 
                              labels=high_degree_nodes,
                              font_size=8)
        
        plt.title(f"Crew Community Graph\n{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges", 
                 fontsize=14)
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma), 
                    label='In-Degree (Influence)', shrink=0.8)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Community graph visualization saved as {filename}")
        plt.close()  # Close to free memory
    
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
    
    def find_optimal_clusters_elbow(self, features: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        print("Finding optimal number of clusters using elbow method...")
        
        if len(features) < 2:
            return 1
        
        # Limit max clusters to reasonable range
        max_clusters = min(max_clusters, len(features) // 2, 10)
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            if k > len(features):
                break
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)
                    
            except Exception as e:
                print(f"Error with k={k}: {e}")
                break
        
        if not inertias:
            return 2  # Default fallback
        
        # Find elbow using rate of change
        optimal_k = 2
        if len(inertias) >= 3:
            # Calculate second derivative to find elbow
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            
            if len(second_diffs) > 0:
                # Find the point where the rate of decrease slows down most
                elbow_idx = np.argmax(second_diffs) + 2  # +2 because we start from k=2
                optimal_k = elbow_idx
        
        # Validate with silhouette score
        if silhouette_scores and len(silhouette_scores) > 0:
            best_silhouette_idx = np.argmax(silhouette_scores)
            best_silhouette_k = best_silhouette_idx + 2  # +2 because we start from k=2
            
            # Use silhouette if it's reasonably close to elbow method
            if abs(best_silhouette_k - optimal_k) <= 1:
                optimal_k = best_silhouette_k
        
        optimal_k = max(2, min(optimal_k, max_clusters))  # Ensure reasonable range
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Inertias: {inertias}")
        if silhouette_scores:
            print(f"Silhouette scores: {[round(s, 3) for s in silhouette_scores]}")
        
        return optimal_k
    
    def assign_levels_with_knn_clustering(self, composite_scores: Dict[str, float]) -> Dict[str, int]:
        """Assign crew levels using KNN clustering with improved balance."""
        print("Assigning crew levels using improved KNN clustering...")
        
        if not composite_scores:
            return {}
        
        # Prepare data for clustering
        user_ids = list(composite_scores.keys())
        scores = np.array(list(composite_scores.values())).reshape(-1, 1)
        
        # Start with 5 clusters (desired levels) and adjust if needed
        target_clusters = 5
        min_cluster_size = max(1, len(user_ids) // 10)  # Minimum 10% of users per cluster
        
        # Find optimal number of clusters between 3-7
        optimal_clusters = self.find_optimal_clusters_elbow(scores, max_clusters=7)
        
        # Use target clusters if optimal is too extreme
        if optimal_clusters < 3:
            optimal_clusters = 3
        elif optimal_clusters > 6:
            optimal_clusters = 5
        
        # Perform K-means clustering
        try:
            # Try multiple random states to get better clustering
            best_kmeans = None
            best_silhouette = -1
            
            for random_state in [42, 123, 456, 789, 999]:
                try:
                    kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_state, n_init=20)
                    cluster_labels = kmeans.fit_predict(scores)
                    
                    # Check cluster balance
                    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                    min_count = np.min(counts)
                    
                    if min_count >= min_cluster_size:
                        if len(unique_labels) > 1:
                            silhouette_avg = silhouette_score(scores, cluster_labels)
                            if silhouette_avg > best_silhouette:
                                best_silhouette = silhouette_avg
                                best_kmeans = kmeans
                except:
                    continue
            
            if best_kmeans is None:
                # Fallback to single attempt
                best_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=20)
                cluster_labels = best_kmeans.fit_predict(scores)
            else:
                cluster_labels = best_kmeans.labels_
            
            cluster_centers = best_kmeans.cluster_centers_.flatten()
            
            # Sort clusters by their center scores (highest to lowest)
            sorted_cluster_indices = np.argsort(cluster_centers)[::-1]
            
            # Create mapping from cluster label to crew level
            cluster_to_level = {}
            for level, cluster_idx in enumerate(sorted_cluster_indices, 1):
                cluster_to_level[cluster_idx] = level
            
            # Assign levels to users
            user_levels = {}
            level_counts = {}
            
            for i, user_id in enumerate(user_ids):
                cluster = cluster_labels[i]
                level = cluster_to_level[cluster]
                user_levels[user_id] = level
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Post-process to ensure reasonable distribution if clusters are too unbalanced
            if len(level_counts) < 3 or max(level_counts.values()) > len(user_ids) * 0.8:
                print("Clustering too unbalanced, applying hybrid approach...")
                return self._apply_hybrid_clustering(composite_scores, optimal_clusters)
            
            print(f"KNN Clustering completed with {optimal_clusters} clusters")
            print("Level distribution:")
            for level in sorted(level_counts.keys()):
                print(f"  Level {level}: {level_counts[level]} users")
            
            print(f"Cluster centers (scores): {sorted(cluster_centers, reverse=True)}")
            print(f"Silhouette score: {best_silhouette:.3f}")
            
            return user_levels
            
        except Exception as e:
            print(f"Error in KNN clustering: {e}")
            return self._apply_hybrid_clustering(composite_scores, 5)
    
    def _apply_hybrid_clustering(self, composite_scores: Dict[str, float], target_clusters: int = 5) -> Dict[str, int]:
        """Apply hybrid clustering combining KMeans with percentile-based adjustment."""
        print("Applying hybrid clustering approach...")
        
        # Sort users by score
        sorted_users = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        total_users = len(sorted_users)
        
        # Use quantile-based initial assignment
        users_per_level = max(1, total_users // target_clusters)
        remainder = total_users % target_clusters
        
        user_levels = {}
        level_counts = {}
        
        current_idx = 0
        for level in range(1, target_clusters + 1):
            # Add extra users to lower levels (better performance gets lower level numbers)
            level_size = users_per_level + (1 if level <= remainder else 0)
            
            for i in range(level_size):
                if current_idx < total_users:
                    user_id, score = sorted_users[current_idx]
                    user_levels[user_id] = level
                    level_counts[level] = level_counts.get(level, 0) + 1
                    current_idx += 1
        
        print("Hybrid clustering completed")
        print("Level distribution:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} users")
        
        return user_levels

if __name__ == "__main__":
    calculator = StandaloneCrewLevelCalculator()
    results_df = calculator.calculate_crew_levels()
    
    if not results_df.empty:
        # Save results
        output_file = "crew_levels_revised.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
    else:
        print("No results to save")
