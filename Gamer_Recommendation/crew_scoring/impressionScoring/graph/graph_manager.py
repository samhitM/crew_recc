"""
Graph operations for impression scoring.
"""
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List
from database.db_manager import DatabaseManager


class GraphManager:
    """Manages graph construction and metrics calculation."""
    
    def __init__(self):
        self.graph = None
        self.pagerank_scores = {}
        self.k_shell_scores = {}
        self.db_manager = DatabaseManager()
    
    def build_friendship_graph(self) -> nx.DiGraph:
        """Build a directed weighted graph from friendship data and user interactions."""
        print("Building directed weighted friendship graph with user interactions...")
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
        print(f"Directed weighted graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
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
    
    def calculate_graph_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate PageRank and K-Shell for all users in directed weighted graph."""
        if self.graph is None:
            self.build_friendship_graph()
        
        print("Calculating graph metrics for directed weighted graph...")
        
        if self.graph.number_of_nodes() == 0:
            print("Empty graph, cannot calculate metrics")
            return {}
        
        # Remove self-loops
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        
        # PageRank for directed weighted graph
        try:
            self.pagerank_scores = nx.pagerank(self.graph, alpha=0.85, max_iter=100, weight='weight')
        except:
            self.pagerank_scores = {node: 1/self.graph.number_of_nodes() for node in self.graph.nodes()}
        
        # K-Shell decomposition (convert to undirected for this metric)
        try:
            undirected_graph = self.graph.to_undirected()
            self.k_shell_scores = nx.core_number(undirected_graph)
        except:
            self.k_shell_scores = {node: 1 for node in self.graph.nodes()}
        
        # Combine metrics (no out_degree anymore)
        all_users = set(self.pagerank_scores.keys()) | set(self.k_shell_scores.keys())
        
        graph_metrics = {}
        for user in all_users:
            graph_metrics[user] = {
                'pagerank': self.pagerank_scores.get(user, 0.0),
                'k_shell': self.k_shell_scores.get(user, 0)
            }
        
        print(f"Calculated metrics for {len(graph_metrics)} users")
        return graph_metrics
    
    def visualize_graph(self, filename: str = "friendship_graph.png"):
        """Visualize the graph and save as PNG."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("No graph to visualize")
            return
        
        print(f"Visualizing graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges...")
        
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
            
            # Color based on PageRank if available
            if hasattr(self, 'pagerank_scores') and node in self.pagerank_scores:
                node_colors.append(self.pagerank_scores[node])
            else:
                node_colors.append(0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_size=node_sizes, 
                             node_color=node_colors,
                             cmap=plt.cm.viridis,
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
        
        plt.title(f"Crew Friendship Graph\n{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges", 
                 fontsize=14)
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                    label='PageRank Score', shrink=0.8)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved as {filename}")
        plt.close()  # Close to free memory
