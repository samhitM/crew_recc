import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from datetime import datetime
from typing import Dict, List
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from database import fetch_all_users_data, get_interaction_type
from ..config.user_data_fields import fields_to_fetch
from levelScoring.config.query_config import QueryConfig
from levelScoring.config.constants import DEFAULT_VALUES


class UserDataFetcher:

    @staticmethod
    def organize_fields_by_table() -> Dict[str, List[str]]:
        """
        Organizes fields by the database table for efficient querying.

        Returns:
        - Dict[str, List[Tuple[str, str]]]: Dictionary mapping tables to their fields and columns.
        """
        table_to_fields = {}
        for field, (column, table) in fields_to_fetch.items():
            if table not in table_to_fields:
                table_to_fields[table] = []
            table_to_fields[table].append((field, column))
        return table_to_fields

    @staticmethod
    def initialize_default_user_data(user_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Initializes user data with default values for all fields.

        Parameters:
        - user_ids (List[str]): List of user IDs.

        Returns:
        - Dict[str, Dict[str, float]]: Dictionary of user data initialized with default values.
        """
        return {
            user_id: {field: DEFAULT_VALUES[field] for field in fields_to_fetch}
            for user_id in user_ids
        }

    @staticmethod
    def fetch_user_data(user_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches user data from the database and handles missing or special cases.

        Parameters:
        - user_ids (List[str]): List of user IDs.

        Returns:
        - Dict[str, Dict[str, float]]: Fetched user data.
        """
        table_to_fields = UserDataFetcher.organize_fields_by_table()
        all_user_data = UserDataFetcher.initialize_default_user_data(user_ids)

        for table, fields in table_to_fields.items():
            columns = ["user_id"] + [col for _, col in fields]
            query_config = QueryConfig.get_query_for_table(table, user_ids)
            
            records = fetch_all_users_data(
                table=query_config['table'],
                database_name=query_config['database_name'],
                columns=columns,
                conditions=query_config['conditions']
            )

            for record in records:
                user_id = record["user_id"]
                for field, column in fields:
                    value = record.get(column, DEFAULT_VALUES[field])

                    # Handle missing and special cases
                    value = UserDataFetcher.replace_missing_values(value, field)
                    value = UserDataFetcher.handle_special_cases(value, field)

                    all_user_data[user_id][field] = value

        return all_user_data

    @staticmethod
    def replace_missing_values(value, field) -> float:
        """
        Replaces missing values with defaults for a given field.

        Parameters:
        - value: The current value.
        - field (str): The field name to check for missing values.

        Returns:
        - float: The value after replacing missing values.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return DEFAULT_VALUES[field]
        return value

    @staticmethod
    def handle_special_cases(value, field) -> float:
        """
        Handles special cases like converting datetime to timestamp.

        Parameters:
        - value: The current value.
        - field (str): The field name to check for special cases.

        Returns:
        - float: The value after handling special cases.
        """
        if field == 'Longevity' and isinstance(value, datetime):
            return value.timestamp()
        return value

    @staticmethod
    def fetch_user_tiers(user_ids: List[str]) -> Dict[str, List[str]]:
        """
        Fetches user tiers from the database.

        Parameters:
        - user_ids (List[str]): List of user IDs.

        Returns:
        - Dict[str, List[str]]: Dictionary mapping user IDs to their tiers.
        """
        query_config = QueryConfig.get_query_for_table("user_tiers", user_ids)
        user_tiers_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions']
        )
        return {record["id"]: record.get("user_tiers", []) for record in user_tiers_data}

    @staticmethod
    def fetch_user_product_data(user_ids: List[str]) -> Dict[str, str]:
        """
        Fetches product data for each user.

        Parameters:
        - user_ids (List[str]): List of user IDs.

        Returns:
        - Dict[str, str]: Dictionary mapping user IDs to product IDs.
        """
        query_config = QueryConfig.get_query_for_table("user_product", user_ids)
        product_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions']
        )
        return {record["user_id"]: record["product_id"] for record in product_data}

    @staticmethod
    def fetch_product_details(product_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches detailed product information for a list of product IDs.

        Parameters:
        - product_ids (List[str]): List of product IDs.

        Returns:
        - Dict[str, Dict[str, float]]: Dictionary mapping product IDs to their details.
        """
        query_config = QueryConfig.get_query_for_table("product", product_ids)
        product_details_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions']
        )
        return {record["product_id"]: record for record in product_details_data}

    @staticmethod
    def fetch_engagement_data(user_ids: List[str]) -> Dict[str, int]:
        """
        Fetches cross-platform engagement data for each user.

        Parameters:
        - user_ids (List[str]): List of user IDs.

        Returns:
        - Dict[str, int]: Dictionary mapping user IDs to engagement counts.
        """
        query_config = QueryConfig.get_query_for_table("user_connections", user_ids)
        engagement_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions']
        )

        engagement_map = {user_id: 0 for user_id in user_ids}  # Default to 0
        for record in engagement_data:
            user_id = record["user_id"]
            if user_id in engagement_map:
                engagement_map[user_id] += 1  # Increment for each unique platform

        return engagement_map
    
    @staticmethod
    def fetch_profile_likes(user_ids: List[str]) -> Dict[str, float]:
        """
        Fetches the number of profile likes for a list of users based on user interactions.

        Parameters:
            user_ids (List[str]): A list of user IDs for which profile likes need to be retrieved.

        Returns:
            Dict[str, float]: A dictionary mapping each user ID to their profile like count.
                            Defaults to a predefined value if no interactions are found.
        """
        
        profile_likes_data = {user_id: DEFAULT_VALUES["Profile_Likes"] for user_id in user_ids}

        # Fetch query configuration for user interactions
        query_config = QueryConfig.get_query_for_table("user_interactions", user_ids)
        results = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns'],
            conditions=query_config['conditions']
        )

        interactions = {user_id: [] for user_id in user_ids}  # Initialize with empty lists
        for result in results:
            entity_id_primary = result["entity_id_primary"]
            user_id = result["user_id"]
            interactions[entity_id_primary].append(user_id)

        for user_id, interacting_users in interactions.items():
            for other_user_id in interacting_users:
                interaction = get_interaction_type(other_user_id, user_id)
                if interaction.get("action") == "like":
                    profile_likes_data[user_id] += 1

        return profile_likes_data

    @staticmethod
    def enrich_user_data_with_additional_info(user_ids: List[str], all_user_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Enriches user data with additional fields like user tiers, rewards, and engagement data.

        Parameters:
        - user_ids (List[str]): List of user IDs.
        - all_user_data (Dict[str, Dict[str, float]]): Dictionary of base user data.

        Returns:
        - Dict[str, Dict[str, float]]: Updated user data with additional fields.
        """
        user_tiers_map = UserDataFetcher.fetch_user_tiers(user_ids)
        product_map = UserDataFetcher.fetch_user_product_data(user_ids)
        product_ids = list(set(product_map.values()))
        product_details_map = UserDataFetcher.fetch_product_details(product_ids)
        engagement_map = UserDataFetcher.fetch_engagement_data(user_ids)
        profile_likes = UserDataFetcher.fetch_profile_likes(user_ids)

        for user_id in user_ids:
            all_user_data[user_id]["Verified_Status"] = 1.0 if 'pro' in user_tiers_map.get(user_id, []) else 0.0
            all_user_data[user_id]["Cross_Platform_Engagement"] = engagement_map.get(user_id, 0)
            all_user_data[user_id]["Profile_Likes"] = profile_likes.get(user_id, DEFAULT_VALUES["Profile_Likes"])

            product_id = product_map.get(user_id, None)
            product_details = product_details_map.get(product_id, {}) if product_id else {}

            if product_details.get("status") == "available": # If product type is SUBSCRIPTION, apply a higher bonus
                all_user_data[user_id]["Rewards"] = 0.2 if product_details.get("product_type") == "SUBSCRIPTION" else 0.05 
            else:
                all_user_data[user_id]["Rewards"] = 0.0 # No product linked, no bonus

        return all_user_data

    @staticmethod
    def normalize_user_data(all_users_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Normalizes all user data fields to the range [0, 1] across all users.

        Parameters:
        - all_users_data (Dict[str, Dict[str, float]]): Dictionary containing raw user data.

        Returns:
        - Dict[str, Dict[str, float]]: Dictionary containing normalized user data.
        """
        field_values = {field: [] for field in next(iter(all_users_data.values())).keys()}

        for user_id, user_data in all_users_data.items():
            for field, value in user_data.items():
                field_values[field].append(value)

        scaler = MinMaxScaler()
        normalized_data = {}

        for field, values in field_values.items():
            values = np.array(values).reshape(-1, 1)
            normalized_values = scaler.fit_transform(values).flatten()

            for idx, user_id in enumerate(all_users_data.keys()):
                if user_id not in normalized_data:
                    normalized_data[user_id] = {}
                normalized_data[user_id][field] = normalized_values[idx]

        return normalized_data