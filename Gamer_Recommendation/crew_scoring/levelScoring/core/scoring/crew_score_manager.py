from levelScoring.utils import UserDataFetcher
from .score_calculator import ScoreCalculator
from database import fetch_all_user_ids

# Main class to manage all users and scores
class CrewScoreManager:
    """
    A class to manage user data, normalize it, and calculate scores for users.
    """

    def __init__(self):
        """
        Initializes the CrewScoreManager with empty user data dictionaries.
        """
        self.all_users_data = {}
        self.normalized_user_data = {}

    def fetch_and_normalize_all_data(self):
        """
        Fetches data for all users and normalizes it.

        This method:
        1. Retrieves all user IDs from the database.
        2. Fetches raw data for each user using the UserDataFetcher.
        3. Normalizes the data across all users using UserDataFetcher.normalize_all_users_data.

        Returns:
            None
        """
        # Fetch all user IDs

        all_user_ids = fetch_all_user_ids()[:3]
        if not all_user_ids:
            print("No user IDs found.")
            return

        # Step 1: Fetch bulk user data
        self.all_users_data = UserDataFetcher.fetch_user_data(all_user_ids)

        # Step 2: Fetch additional data like tiers, rewards, and engagement
        self.all_users_data = UserDataFetcher.enrich_user_data_with_additional_info(all_user_ids, self.all_users_data)
    
        # Step 3: Normalize user data
        self.normalized_user_data = UserDataFetcher.normalize_user_data(self.all_users_data)
        print(self.normalized_user_data)

    def get_user_score(self, user_id: str) -> float:
        """
        Calculates and returns the composite score for a specified user.

        Parameters:
            user_id (str): The unique identifier for the user.

        Returns:
            float: The composite score for the user. If data is not fetched
                   or the user ID is not found, returns 0.0.
        """
        if not self.normalized_user_data:
            print("Data has not been fetched and normalized. Please fetch and normalize the data first.")
            return 0.0

        if user_id not in self.normalized_user_data:
            print(f"User ID {user_id} not found in normalized data.")
            return 0.0

        # Calculate and return composite score for the given user
        user_data = self.normalized_user_data[user_id]
        return ScoreCalculator.calculate_composite_score(user_data)
