from datetime import datetime
from typing import Dict
from core.database import fetch_db_user_value
from levelScoring.constants import DEFAULT_VALUES
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Data Fetcher Class
class UserDataFetcher:
    """
    Fetches and processes user data from the database, providing normalized user data
    for further analysis and scoring.
    """
    @staticmethod
    def fetch_user_data(user_id: str) -> Dict[str, float]:
        """
        Fetches raw user data from the database for a given user_id.

        Parameters:
        - user_id (str): Unique identifier for the user.

        Returns:
        - Dict[str, float]: A dictionary containing the user's data, with field names as keys
                            and their corresponding values. Defaults are used for missing or
                            unavailable data.
        """
        
        fields_to_fetch = {
            'Max_Hours': ('crew_online_time', 'crew_user'),
            'Achievements': ('crew_badge', 'crew_user'), # May have to modify this
            'Total_Impression_Score': ('crew_impression', 'crew_user'),
            'Consistent_Engagement': ('total_active_sec', 'crew_user'), #Change it to user_sessions table later
            'Longevity': ('created_ts', 'crew_user'),
            'Event_Participation': ('event_participation', 'crew_user'), # Modify the table names
            'Community_Contributions': ('contributions', 'crew_user'),
            'Social_Interactions': ('social_interactions', 'crew_user'),
            'Cross_Platform_Engagement': ('cross_platform', 'crew_user'),
        }

        user_data = {}
        for field, (column, table) in fields_to_fetch.items():
            try:
                value = fetch_db_user_value(user_id, column, table)
                if value is None:
                    value = DEFAULT_VALUES[field]
                # Convert `created_ts` (Longevity) to timestamp
                if field == 'Longevity' and isinstance(value, datetime):
                    value = value.timestamp()  # Convert to float (seconds since epoch)
                user_data[field] = value
            except Exception as e:
                print(f"Error fetching {field} for User ID {user_id}: {str(e)}. Using default value: {DEFAULT_VALUES[field]}")
                user_data[field] = DEFAULT_VALUES[field]

        # Fetch and verify user tiers for additional weighting
        try:
            user_tiers = fetch_db_user_value(user_id, 'user_tiers', 'user',id_field="id", default_value=[])
            if isinstance(user_tiers, list) and ('pro' in user_tiers or 'elite' in user_tiers):
                user_data['Verified_Status'] = 1.0  # Verified
            else:
                user_data['Verified_Status'] = 0.0  # Not Verified
        except Exception as e:
            print(f"Error fetching user_tiers for User ID {user_id}: {str(e)}. Defaulting Verified_Status to 0.")
            user_data['Verified_Status'] = 0.0

        return user_data

    @staticmethod
    def normalize_all_user_data(all_user_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Normalizes all user data fields to the range [0, 1] across all users.

        Parameters:
        - all_user_data (Dict[str, Dict[str, float]]): Dictionary containing raw user data
                                                       for all users.

        Returns:
        - Dict[str, Dict[str, float]]: Dictionary containing normalized user data for each user.
        """
        # Collect all values for each field across all users for normalization
        field_values = {field: [] for field in list(next(iter(all_user_data.values())).keys())}

        # Collect field values from all users
        for user_id, user_data in all_user_data.items():
            for field, value in user_data.items():
                field_values[field].append(value)

        # Normalize each field using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = {}

        for field, values in field_values.items():
            values = np.array(values).reshape(-1, 1)
            normalized_values = scaler.fit_transform(values).flatten()

            # Assign normalized values back to users
            for idx, user_id in enumerate(all_user_data.keys()):
                if user_id not in normalized_data:
                    normalized_data[user_id] = {}
                normalized_data[user_id][field] = normalized_values[idx]
        
        print(normalized_data)
        return normalized_data
