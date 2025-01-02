import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from core.database import get_user_message_stats

import warnings
from requests.exceptions import RequestException

from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

class CrewImpressions:
    def __init__(self, api_url, jwt_token):
        """
        Initialize the CrewImpressions class with the API URL and JWT token.

        Parameters:
            api_url (str): Base API URL.
            jwt_token (str): JWT token for authentication.
        """
        self.api_url = api_url
        self.jwt_token = jwt_token
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        self.graph = nx.Graph()

    def fetch_user_data(self, user_id):
        """
        Fetch user data from multiple endpoints and map attributes to a user-friendly format.

        Parameters:
            user_id (str): The ID of the user whose data is to be fetched.

        Returns:
            dict: Mapped user data.
        """
        profile_details_url = f"{self.api_url}/profile-details"
        profile_stats_url = f"{self.api_url}/profile-stats"

        user_data = {
            "User_ID": user_id,
            "K_Shell": None,
            "Out_Degree": None,
            "Reposts": np.random.randint(0, 100),
            "Replies": None,
            "Mentions": None,
            "Favorites": None,
            "Interest_Topic": np.random.uniform(0, 1),
            "Bio_Content": None,
            "Profile_Likes": np.random.randint(0, 500),
            "User_Games": None,
            "Verified_Status": None,
            "Posts_on_Topic": np.random.randint(0, 100),
            "Bonus": np.random.uniform(0, 1),
            "Unique_Pageviews": np.random.randint(10, 500),
            "Scroll_Depth_Percent": np.random.uniform(10, 100),
            "Timestamp": None,
            "Impressions": None,
            "Engagement_Per_Impression": np.random.uniform(0.1, 1.0),
            "Total_Messages": 0  # Add field for total messages
        }

        try:
            # Fetch `/profile-details` data
            profile_details_response = requests.get(
                profile_details_url,
                headers=self.headers,
                verify=False,  # Disable SSL verification Comment out this line later on
                timeout=10
            )
            if profile_details_response.status_code == 200:
                profile_details = profile_details_response.json()
                user_data.update({
                    "Out_Degree": profile_details.get("total_friends", 0),
                    "Bio_Content": 1 if profile_details.get("description") else 0,
                    "User_Games": profile_details.get("total_games", 0),
                    "Verified_Status": 1 if profile_details.get("crew_badge") else 0,
                    "Timestamp": profile_details.get("register_ts"),
                })

            # Fetch `/profile-stats` data
            profile_stats_response = requests.get(
                profile_stats_url,
                headers=self.headers,
                verify=False,  # Disable SSL verification Comment out this line later on
                timeout=10
            )
            if profile_stats_response.status_code == 200:
                profile_stats = profile_stats_response.json()
                user_data.update({
                    "Impressions": profile_stats.get("crew_impression", 0) if profile_stats and profile_stats.get("crew_impression") is not None else 0,
                    "Replies": profile_stats.get("player_stats", {}).get("replies", 0) if profile_stats and isinstance(profile_stats.get("player_stats"), dict) else 0,
                    "Mentions": profile_stats.get("player_stats", {}).get("mentions", 0) if profile_stats and isinstance(profile_stats.get("player_stats"), dict) else 0,
                    "Favorites": profile_stats.get("player_stats", {}).get("favorites", 0) if profile_stats and isinstance(profile_stats.get("player_stats"), dict) else 0,
                })
            
            # Fetch total messages using get_user_message_stats
            message_stats = get_user_message_stats(user_id)
            user_data["Total_Messages"] = sum(topic["totalMessages"] for topic in message_stats)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for user {user_id}: {e}")

        return user_data

    def fetch_user_relations(self, user_id=None, limit=50, offset=0, relation="pending"):
        """
        Fetch user relations from the API with optional pagination and filtering.

        Parameters:
            user_id (str): The ID of the user whose relations are to be fetched.
            limit (int): Maximum number of relations to return.
            offset (int): Number of relations to skip.
            relation (str): Type of relation to filter.

        Returns:
            list: List of user relations.
        """
        params = {
            "limit": limit,
            "offset": offset,
        }
        data = {"relation": relation}
        
        if user_id is not None:
            data["user_id"] = user_id

        try:
            response = requests.get(
                f"{self.api_url}/user/relations",
                headers=self.headers,
                params=params,
                json=data,
                verify=False,  # Disable SSL verification Comment out this line later on
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return []  # No relations found
            else:
                raise ValueError(f"Unexpected status code: {response.status_code}")
        except RequestException as e:
            raise ValueError(f"Error fetching relations: {str(e)}")

    def build_graph(self):
        """
        Build a graph based on user relations fetched from the API.

        Returns:
            nx.Graph: A graph representing user relations.
        """
        visited_users = set()
        user_queue = []

        # Fetch initial user relations
        initial_relations = self.fetch_user_relations()
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
            relations = self.fetch_user_relations(user_id=current_user)
            for relation in relations:
                friend_id = relation["user_id"]
                self.graph.add_node(friend_id)
                self.graph.add_edge(current_user, friend_id)

                if friend_id not in visited_users:
                    user_queue.append(friend_id)

        return self.graph

    def calculate_k_shell(self):
        """
        Calculate the K-shell decomposition for the graph.

        Returns:
            dict: K-shell values for each user.
        """
        if not self.graph:
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
        user_data = self.fetch_user_data(user_id)
        k_shell_values = self.calculate_k_shell()
        user_data["K_Shell"] = k_shell_values.get(user_id, 0)
        return user_data


class CrewScoreCalculator:
    def __init__(self, crew_impressions_instance):
        self.crew_impressions = crew_impressions_instance

    def calculate_topological_score(self, user_data, k_shell_weight=0.6, out_degree_weight=0.4):
        k_shell_score = user_data['K_Shell'] * k_shell_weight
        out_degree_score = user_data['Out_Degree'] * out_degree_weight
        return k_shell_score + out_degree_score

    def calculate_user_feature_score(self, user_data, feature_weights):
        features = [
            'Reposts', 'Replies', 'Mentions', 'Favorites', "Total_Messages", 'Interest_Topic',
            'Bio_Content', 'Profile_Likes', 'User_Games', 'Verified_Status', 'Posts_on_Topic'
        ]
        score = sum(user_data[feature] * weight for feature, weight in zip(features, feature_weights))
        return score

    def calculate_total_score(self, topological_score, user_feature_score, website_impressions, beta=0.5, gamma=0.3, bonus=0):
        return beta * topological_score + gamma * user_feature_score + (1 - beta - gamma) * website_impressions + bonus

    def fetch_and_calculate_scores(self, user_id):
        user_data = self.crew_impressions.fetch_complete_user_data(user_id)

        feature_weights = [0.25, 0.15, 0.2, 0.1, 0.02, 0.2, 0.05, 0.05, 0.05, 0.03, 0.05]
        topological_score = self.calculate_topological_score(user_data)
        user_feature_score = self.calculate_user_feature_score(user_data, feature_weights)
        website_impressions = user_data['Unique_Pageviews'] * (1 + user_data['Scroll_Depth_Percent'] / 100)
        total_score = self.calculate_total_score(
            topological_score, user_feature_score, website_impressions, bonus=user_data['Bonus']
        )

        user_data.update({
            'Topological_Score': topological_score,
            'User_Feature_Score': user_feature_score,
            'Website_Impressions': website_impressions,
            'Total_Score': total_score,
        })

        return user_data

    def aggregate_impressions(self, user_data_list):
        df = pd.DataFrame(user_data_list)

        # Convert to datetime and handle timezone explicitly
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True).dt.tz_localize(None)

        # Extract date-based features
        df['Date'] = df['Timestamp'].dt.date
        df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
        df['Month'] = df['Timestamp'].dt.to_period('M').apply(lambda r: r.start_time)

        # Aggregate impressions
        daily_impressions = df.groupby('Date')['Impressions'].sum()
        weekly_impressions = df.groupby('Week')['Impressions'].sum()
        monthly_impressions = df.groupby('Month')['Impressions'].sum()

        return {
            'Daily_Impressions': daily_impressions,
            'Weekly_Impressions': weekly_impressions,
            'Monthly_Impressions': monthly_impressions
        }


    def plot_distributions(self, user_data_list, user_ids):
        df = pd.DataFrame(user_data_list)
        df = df[df['User_ID'].isin(user_ids)]
        
        plt.figure(figsize=(12, 10))
        for i, score_type in enumerate(['Topological_Score', 'User_Feature_Score', 'Website_Impressions', 'Total_Score']):
            plt.subplot(2, 2, i + 1)
            plt.hist(df[score_type], bins=30, alpha=0.7, label=score_type, color=f'C{i}')
            plt.title(f"Distribution of {score_type}")
            plt.xlabel("Scores")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()
        plt.suptitle("Score Distributions")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    

if __name__ == "__main__":
    api_url = "https://localhost:3000/api"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4UHpJOG5FUXU1TCIsImVtYWlsIjoieWFzaHlhZGF2MDBAZmxhc2guY28iLCJpYXQiOjE3MzIxOTE3MjYsImV4cCI6MTczNzM3NTcyNn0.DeRiGzNUflr6_8CSqrw3K7UkybEb8pJe9ocD9Gs5Axs"
    user_ids = ["3oYkVCJdEag", "dCUKB2Vf9Zk", "User_3"]

    crew_impressions = CrewImpressions(api_url, jwt_token)
    # Build the graph
    print("Building graph...")
    crew_impressions.build_graph()

    # Fetch complete user data with K-shell
    print("Fetching complete user data...")
    complete_user_data = crew_impressions.fetch_complete_user_data(user_ids[0])
    print("Complete User Data:")
    print(complete_user_data)
    
    crew_score_calculator = CrewScoreCalculator(crew_impressions)

    user_data_list = [crew_score_calculator.fetch_and_calculate_scores(user_id) for user_id in user_ids]
    # print(user_data_list)
    # Aggregate impressions
    aggregated_data = crew_score_calculator.aggregate_impressions(user_data_list)
    print("Aggregated Impressions:")
    print(aggregated_data)

    # Plot distributions
    crew_score_calculator.plot_distributions(user_data_list, user_ids)