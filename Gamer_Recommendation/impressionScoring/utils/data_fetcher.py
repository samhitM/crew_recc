import requests
from core.database import get_user_message_stats
import warnings
from requests.exceptions import RequestException
import numpy as np
from urllib3.exceptions import InsecureRequestWarning
from core.database import fetch_db_user_value
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

class DataFetcher:
    def __init__(self, api_url, jwt_token):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {jwt_token}"}

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
            "Total_Messages": 0  
        }

        try:
            # Fetch `/profile-details` data
            profile_details_response = requests.get(
                profile_details_url,
                headers=self.headers,
                verify=False,  # Disable SSL verification Comment out this line later on
                timeout=10
            )
            user_tiers = fetch_db_user_value(user_id, 'user_tiers', 'user', id_field="id", default_value=[])
            verified_status = 1.0 if isinstance(user_tiers, list) and ('pro' in user_tiers or 'elite' in user_tiers) else 0.0

            if profile_details_response.status_code == 200:
                profile_details = profile_details_response.json()
                user_data.update({
                    "Out_Degree": profile_details.get("total_friends", 0),
                    "Bio_Content": 1 if profile_details.get("description") else 0,
                    "User_Games": profile_details.get("total_games", 0),
                    "Verified_Status": verified_status,
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
            topics = message_stats["userStats"]  # Extract the list of topics
            user_data["Total_Messages"] = sum(topic["totalMessages"] for topic in topics)
            
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