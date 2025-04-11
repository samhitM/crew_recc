import networkx as nx
from crew_scoring.impressionScoring.utils import DataFetcher
from sklearn.preprocessing import MinMaxScaler
from services.recommendation_utils import get_user_id_to_username_mapping
import numpy as np

class CrewScoreManager:
    """
    A manager class to handle Crew Scores and user relations using a graph-based approach.
    """
    
    def __init__(self, api_url, user_ids):
        """
        Initialize the CrewScoreManager class with the API URL, JWT token, and user IDs.

        Parameters:
            api_url (str): Base API URL.
            jwt_token (str): JWT token for authentication.
            user_ids (list): List of user IDs for which the graph and data need to be processed.
        """
        self.api_url = api_url
        self.graph = None  # Initialize the graph lazily
        self.user_ids = user_ids
        self.data_fetcher = DataFetcher(api_url=api_url, user_ids=user_ids)
        self.user_id_to_username = get_user_id_to_username_mapping(self.user_ids, name_field="username") # Get mapping of user_id to username


    def build_graph(self):
        """
        Build a full graph by fetching relations for all users in self.user_ids.

        Returns:
            nx.Graph: A graph representing user relations using usernames.
        """
        if self.graph is not None:
            return self.graph  # Already built

        self.graph = nx.Graph()
        try:
            for user_id in self.user_ids:
                # Fetch all friends of this user
                relations = self.data_fetcher.fetch_user_relations(user_id=user_id)
                username = self.user_id_to_username.get(user_id, user_id)

                self.graph.add_node(username)

                for relation in relations:
                    friend_username = relation.get("username")  # use username directly from relation
                    if friend_username:
                        self.graph.add_node(friend_username)
                        self.graph.add_edge(username, friend_username)

        except Exception as e:
            raise RuntimeError(f"Failed to build graph: {e}")
        
        # print("Nodes:", list(self.graph.nodes))
        # print("Edges:", list(self.graph.edges))
        return self.graph
    
    def calculate_k_shell(self):
        """
        Calculate the K-shell decomposition for the graph.

        Returns:
            dict: K-shell values for each user.
        """
        if self.graph is None:
            raise ValueError("Graph is empty. Build the graph first.")
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        return nx.core_number(self.graph)

    def _normalize_features(self, all_users_data):
        """
        Normalizes all user data fields to the range [0, 1] across all users,
        excluding the 'Posts_created_ts' field.
        """
        # Extract all field names, excluding 'Posts_created_ts'
        sample_user = next(iter(all_users_data.values()))
        fields_to_normalize = [field for field in sample_user.keys() if field != "Posts_created_ts"]
        
        # Collect values for each field
        field_values = {field: [] for field in fields_to_normalize}
        
        for user_data in all_users_data.values():
            for field in fields_to_normalize:
                field_values[field].append(user_data[field])

        # Normalize using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = {}

        for field, values in field_values.items():
            values = np.array(values).reshape(-1, 1)
            normalized_values = scaler.fit_transform(values).flatten()

            for idx, user_id in enumerate(all_users_data.keys()):
                if user_id not in normalized_data:
                    normalized_data[user_id] = {}

                normalized_data[user_id][field] = normalized_values[idx]

        # Retain original 'Posts_created_ts' values
        for user_id, user_data in all_users_data.items():
            normalized_data[user_id]["Posts_created_ts"] = user_data["Posts_created_ts"]

        return normalized_data

    def fetch_complete_user_data(self):
        """
        Fetch complete user data including K-shell decomposition for all user_ids at once.

        Returns:
            dict: Updated user data dictionary with K-shell values.
        """
        if self.graph is None:
            self.build_graph()  # Ensure the graph is built

        try:
            # Fetch all user data at once for user_ids
            all_user_data = self.data_fetcher.fetch_user_data()  # Fetch data for all users
            k_shell_values = self.calculate_k_shell()
            
            # Update each user's data with K-shell value
            for user_id, data in all_user_data.items():
                data["K_Shell"] = k_shell_values.get(self.user_id_to_username.get(user_id), 0)
            
            # Normalize features
            all_user_data = self._normalize_features(all_user_data)
            return all_user_data
        
        except Exception as e:
            raise RuntimeError(f"Failed to fetch complete user data: {e}")