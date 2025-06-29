"""
Graph operations for level scoring.
"""
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List
from database.level_db_manager import LevelDatabaseManager


class LevelGraphManager:
    """Manages graph construction for level scoring."""
    
    def __init__(self):
        self.graph = None
        self.db_manager = LevelDatabaseManager()
    
    def build_community_graph(self) -> nx.DiGraph:
        """Build a directed weighted community graph with user interactions."""
        print("Building directed weighted community graph with user interactions...")
        self.graph = nx.DiGraph()  # Use directed graph
        friendship_data = self.db_manager.fetch_friendship_data()
        interaction_data = self.db_manager.fetch_user_interactions()
        
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
    
    def calculate_link_prediction_scores(self) -> Dict[str, float]:
        """Calculate link prediction scores for level scoring."""
        print("Calculating link prediction scores...")
        
        if self.graph is None:
            self.build_community_graph()
        
        link_scores = {}
        
        # For each user, calculate their link prediction score based on graph structure
        for user in self.graph.nodes():
            # Calculate score based on in-degree, out-degree, and clustering
            in_degree = self.graph.in_degree(user, weight='weight')
            out_degree = self.graph.out_degree(user, weight='weight')
            
            # Try to calculate clustering coefficient
            try:
                clustering = nx.clustering(self.graph.to_undirected(), user, weight='weight')
            except:
                clustering = 0
            
            # Combine metrics for link prediction score
            link_score = (in_degree * 0.4 + out_degree * 0.4 + clustering * 0.2)
            link_scores[user] = link_score
        
        print(f"Calculated link prediction scores for {len(link_scores)} users")
        return link_scores
    
    def visualize_community_graph(self, filename: str = "community_graph.png", communities: Dict[str, int] = None):
        """Visualize the community graph with community circles and save as PNG."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("No graph to visualize")
            return
        
        print(f"Visualizing community graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges...")
        
        plt.figure(figsize=(20, 16))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=3, iterations=100, seed=42)
        
        # If communities are provided, draw community circles
        if communities:
            import matplotlib.patches as patches
            
            # Group nodes by community
            community_groups = {}
            for node, community_id in communities.items():
                if community_id not in community_groups:
                    community_groups[community_id] = []
                community_groups[community_id].append(node)
            
            # Define colors for communities
            colors = plt.cm.Set3(np.linspace(0, 1, len(community_groups)))
            
            # Draw faint circles around each community
            ax = plt.gca()
            for i, (community_id, nodes) in enumerate(community_groups.items()):
                if len(nodes) > 1:  # Only draw circles for communities with multiple members
                    # Get positions of nodes in this community
                    community_pos = [pos[node] for node in nodes if node in pos]
                    if community_pos:
                        # Calculate center and radius of community
                        x_coords = [p[0] for p in community_pos]
                        y_coords = [p[1] for p in community_pos]
                        
                        center_x = np.mean(x_coords)
                        center_y = np.mean(y_coords)
                        
                        # Calculate radius as max distance from center + padding
                        max_distance = max([np.sqrt((x - center_x)**2 + (y - center_y)**2) 
                                          for x, y in community_pos])
                        radius = max_distance + 0.15
                        
                        # Draw faint circle
                        circle = patches.Circle((center_x, center_y), radius, 
                                              linewidth=2, edgecolor=colors[i], 
                                              facecolor=colors[i], alpha=0.15,
                                              linestyle='--')
                        ax.add_patch(circle)
                        
                        # Add community label
                        plt.text(center_x, center_y + radius + 0.05, f'Community {community_id}', 
                               ha='center', va='bottom', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
        
        # Draw nodes with community-based colors
        node_sizes = []
        node_colors = []
        for node in self.graph.nodes():
            # Size based on degree
            degree = self.graph.degree(node)
            node_sizes.append(max(100, degree * 25))
            
            # Color based on community membership
            if communities and node in communities:
                community_id = communities[node]
                color_idx = list(set(communities.values())).index(community_id)
                node_colors.append(color_idx)
            else:
                node_colors.append(-1)  # Gray for non-community members
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_size=node_sizes, 
                             node_color=node_colors,
                             cmap=plt.cm.Set3,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=1)
        
        # Draw edges with weights
        edges = self.graph.edges()
        weights = [self.graph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(self.graph, pos,
                             width=[w * 2 for w in weights],
                             alpha=0.4,
                             edge_color='gray')
        
        # Add labels for all nodes (smaller font)
        nx.draw_networkx_labels(self.graph, pos, 
                              font_size=8, font_weight='bold')
        
        # Create title with community info
        title = f"Crew Community Graph\n{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        if communities:
            num_communities = len(set(communities.values()))
            title += f", {num_communities} communities detected"
        
        plt.title(title, fontsize=16)
        
        # Remove axes
        plt.axis('off')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Community graph visualization saved as {filename}")
        plt.close()  # Close to free memory
