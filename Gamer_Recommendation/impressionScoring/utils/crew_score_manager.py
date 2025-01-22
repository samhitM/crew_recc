import networkx as nx
from .data_fetcher import DataFetcher

class CrewScoreManager:
    """
    A manager class to handle Crew Scores and user relations using a graph-based approach.
    """
    
    def __init__(self, api_url, jwt_token):
        """
        Initialize the CrewImpressions class with the API URL and JWT token.

        Parameters:
            api_url (str): Base API URL.
            jwt_token (str): JWT token for authentication.
        """
        self.api_url = api_url
        self.jwt_token = jwt_token
        self.headers = {"Authorization": f"Bearer {self.jwt_token}"}
        self.graph = None  # Initialize graph lazily
        self.data_fetcher = DataFetcher(api_url=api_url, jwt_token=jwt_token)

    def build_graph(self):
        """
        Build a graph based on user relations fetched from the API.

        Returns:
            nx.Graph: A graph representing user relations.
        """
        if self.graph is not None:
            return self.graph  # Skip if already built

        self.graph = nx.Graph()
        visited_users = set()
        user_queue = []

        try:
            # Fetch initial user relations
            initial_relations = self.data_fetcher.fetch_user_relations()
            for user in initial_relations:
                user_id = user["user_id"]
                self.graph.add_node(user_id)
                user_queue.append(user_id)

            while user_queue:
                current_user = user_queue.pop(0)
                if current_user in visited_users:
                    continue

                visited_users.add(current_user)

                # Fetch friends of the current user
                relations = self.data_fetcher.fetch_user_relations(user_id=current_user)
                for relation in relations:
                    friend_id = relation["user_id"]
                    self.graph.add_node(friend_id)
                    self.graph.add_edge(current_user, friend_id)

                    if friend_id not in visited_users:
                        user_queue.append(friend_id)

        except Exception as e:
            raise RuntimeError(f"Failed to build graph: {e}")

        return self.graph

    def calculate_k_shell(self):
        """
        Calculate the K-shell decomposition for the graph.

        Returns:
            dict: K-shell values for each user.
        """
        if self.graph is None:
            raise ValueError("Graph is empty. Build the graph first.")
        return nx.core_number(self.graph)

    def fetch_complete_user_data(self, user_id):
        """
        Fetch complete user data including K-shell decomposition.

        Parameters:
            user_id (str): The ID of the user.

        Returns:
            dict: Complete user data with K-shell value.
        """
        if self.graph is None:
            self.build_graph()  # Ensure the graph is built

        try:
            user_data = self.data_fetcher.fetch_user_data(user_id)
            k_shell_values = self.calculate_k_shell()
            user_data["K_Shell"] = k_shell_values.get(user_id, 0)
            # user_data["K_Shell"] = 0
            return user_data
        except Exception as e:
            raise RuntimeError(f"Failed to fetch user data for {user_id}: {e}")
