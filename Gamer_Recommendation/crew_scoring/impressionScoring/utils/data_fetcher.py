import requests
import warnings
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from requests.exceptions import RequestException
from urllib3.exceptions import InsecureRequestWarning
from database import (
    get_users_message_stats, 
    fetch_all_users_data, 
)
from typing import List, Dict, Optional
from impressionScoring.config.constants import DEFAULT_VALUES
from impressionScoring.config.query_config import QueryConfig
from datetime import datetime
from services.token_utils import generate_jwt_token

from core.config import VERIFY_SSL


warnings.filterwarnings("ignore", category=InsecureRequestWarning)


class DataFetcher:
    def __init__(self, api_url: str, user_ids: List[str]):
        self.api_url = api_url
        self.user_ids = user_ids  

    def initialize_default_user_data(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize user data with default values for all users.

        Returns:
            Dict[str, Dict[str, float]]: Default user data for each user.
        """
        return {user_id: DEFAULT_VALUES.copy() for user_id in self.user_ids}

    def _make_request(self, url: str, user_id: str, params=None, json_data=None):
        """Helper function to make GET requests."""
        headers = {"Authorization": f"Bearer {generate_jwt_token(user_id)}"}
        try:
            response = requests.get(
                url, headers=headers, params=params, json=json_data, verify=VERIFY_SSL, timeout=10
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                raise ValueError(f"Unexpected status code: {response.status_code}")
        except RequestException as e:
            print(f"Request error: {e}")
            return None

    def _fetch_profile_details(self) -> Dict[str, Dict[str, float]]:
        """Fetch profile details for multiple users from API."""
        
        url = f"{self.api_url}/profile-details"
        profile_data = self.initialize_default_user_data()

        for user_id in self.user_ids:
            response = self._make_request(url,user_id,params={"user_id": user_id})
            if response:
                profile_data[user_id].update({
                    "Out_Degree": response.get("total_friends", DEFAULT_VALUES["Out_Degree"]),
                    "Bio_Content": 1 if response.get("description") else DEFAULT_VALUES["Bio_Content"],
                    "User_Games": response.get("total_games", DEFAULT_VALUES["User_Games"]),
                    "Bonus" : int(datetime.strptime(response.get("register_ts", DEFAULT_VALUES["Bonus"])[:-1], "%Y-%m-%dT%H:%M:%S.%f").timestamp())
                })

        return profile_data

    def _fetch_user_tiers(self) -> Dict[str, float]:
        """
        Fetches user tiers and assigns a tier value based on their status.

        Returns:
            Dict[str, float]: A dictionary mapping user IDs to their respective tier values.
                            - 1.0 if the user has a 'pro' or 'elite' tier.
                            - 0.0 otherwise.
                            If no data is found, a default value is assigned.
        """
        
        user_tiers_data = {user_id: DEFAULT_VALUES["Verified_Status"] for user_id in self.user_ids}

        # Fetch query configuration for the user_tiers table
        query_config = QueryConfig.get_query_for_table("user_tiers", self.user_ids)

        results = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions']
        )

        for result in results:
            user_id = result["id"]
            user_tiers = result.get("user_tiers", [])
            user_tiers_data[user_id] = 1.0 if isinstance(user_tiers, list) and ('pro' in user_tiers or 'elite' in user_tiers) else 0.0

        return user_tiers_data

    def _fetch_user_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Fetches user engagement metrics, including replies, mentions, favorites, 
        and profile likes for multiple users.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where each user ID maps to another dictionary 
                                        containing their respective engagement metrics.
                                        Defaults are used if no data is available.
        """
    
        user_metrics_data = self.initialize_default_user_data()

        # Fetch Replies
        query_config = QueryConfig.get_query_for_table("posts_replies", self.user_ids)
        replies_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions'],
            group_by=query_config.get('group_by', None)
        )
        replies_dict = {item["parent_id"]: item["Replies"] for item in replies_data if item["parent_id"] is not None}

        # Fetch Mentions
        query_config = QueryConfig.get_query_for_table("posts_mentions", self.user_ids)
        mentions_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions'],
            group_by=query_config.get('group_by', None)
        )
        mentions_dict = {item["user_id"]: item["Mentions"] for item in mentions_data if item["user_id"] is not None}

        # Fetch Favorites
        query_config = QueryConfig.get_query_for_table("posts_favorites", self.user_ids)
        favorites_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions'],
            group_by=query_config.get('group_by', None)
        )
        favorites_dict = {item["user_id"]: item["Favorites"] for item in favorites_data if item["user_id"] is not None}

        # Populate user metrics for each user
        for user_id in self.user_ids:
            user_metrics_data[user_id]["Replies"] = replies_dict.get(user_id, DEFAULT_VALUES["Replies"])
            user_metrics_data[user_id]["Mentions"] = mentions_dict.get(user_id, DEFAULT_VALUES["Mentions"])
            user_metrics_data[user_id]["Favorites"] = favorites_dict.get(user_id, DEFAULT_VALUES["Favorites"])

        return user_metrics_data

    def _fetch_total_messages(self) -> Dict[str, int]:
        """
        Fetches the total message count for multiple users in a single query.

        Returns:
            Dict[str, int]: A dictionary mapping user IDs to their total number of messages.
                            If no data is found, a default value is assigned.
        """
    
        total_messages_data = {user_id: DEFAULT_VALUES["Total_Messages"] for user_id in self.user_ids}
        
        # Fetch message stats for all users at once
        message_stats = get_users_message_stats(self.user_ids)
        
        # Aggregate total messages for each user
        for user_id, topics in message_stats.items():
            total_messages_data[user_id] = sum(topic.get("totalMessages", 0) for topic in topics)

        return total_messages_data

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
        
        url = f"{self.api_url}/users/relations"
        params = {"limit": limit, "offset": offset}
        data = {"relation": relation, "user_id": user_id} if user_id else {"relation": relation}
        print("Fetching for:",user_id)
        return self._make_request(url, user_id=user_id, params=params, json_data=data) or []
    

    def _fetch_user_post_timestamps(self) -> Dict[str, Optional[datetime]]:
        """
        Fetches the latest post creation timestamp for each user.

        Returns:
            Dict[str, Optional[datetime]]: A dictionary mapping user IDs to their latest post creation timestamp.
                                        If no posts exist, the value remains None.
        """
    
        user_posts_data = {user_id: None for user_id in self.user_ids}  # Default to None

        # Fetch query configuration for the posts_created_ts table
        query_config = QueryConfig.get_query_for_table("posts_created_ts", self.user_ids)

        results = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions'],
            order_by=query_config['order_by']
        )

        for result in results:
            user_id = result["user_id"]
            created_ts = result.get("created_ts")

            if isinstance(created_ts, str):
                created_ts = datetime.strptime(created_ts, "%Y-%m-%d %H:%M:%S")  # Convert string to datetime

            # Keep the latest timestamp for each user
            if user_id in user_posts_data:
                if user_posts_data[user_id] is None or (created_ts and created_ts > user_posts_data[user_id]):
                    user_posts_data[user_id] = created_ts

        return user_posts_data
    

    def fetch_user_data(self) -> Dict[str, Dict[str, float]]:
        """
        Fetches and aggregates user data from multiple sources into a structured format.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where each user ID maps to another dictionary 
                                        containing various user attributes and metrics.
        """
    
        user_data = self.initialize_default_user_data()
        profile_details = self._fetch_profile_details()
        user_tiers = self._fetch_user_tiers()
        user_metrics = self._fetch_user_metrics()
        total_messages = self._fetch_total_messages()
        posts_created_ts = self._fetch_user_post_timestamps() 

        for user_id in self.user_ids:
            user_data[user_id].update(profile_details.get(user_id, {}))
            
            # Only update Verified_Status if it's missing
            if "Verified_Status" not in user_data[user_id]:
                user_data[user_id]["Verified_Status"] = user_tiers.get(user_id, DEFAULT_VALUES["Verified_Status"])

            # Update other metrics without overwriting valid existing values
            for key, value in user_metrics.get(user_id, {}).items():
                if user_data[user_id].get(key, None) in (None, DEFAULT_VALUES[key]):
                    user_data[user_id][key] = value

            if user_data[user_id]["Total_Messages"] == DEFAULT_VALUES["Total_Messages"]:
                user_data[user_id]["Total_Messages"] = total_messages.get(user_id, DEFAULT_VALUES["Total_Messages"])

            # Update posts_created_ts (latest post timestamp)
            if "Posts_created_ts" not in user_data[user_id] or user_data[user_id]["Posts_created_ts"] is None:
                user_data[user_id]["Posts_created_ts"] = posts_created_ts.get(user_id, None)
        
        return user_data