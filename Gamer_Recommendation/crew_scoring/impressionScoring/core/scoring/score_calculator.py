from crew_scoring.impressionScoring.config.constants import *


class CrewScoreCalculator:
    """
    A class to calculate various scores for users based on topological, feature-based, and website impression metrics.
    """

    def calculate_topological_score(self, user_data):
        """
        Calculate the topological score for a user.

        Parameters:
            user_data (dict): Dictionary containing user information.

        Returns:
            float: The calculated topological score.
        """
        k_shell_score = user_data['K_Shell'] * K_SHELL_WEIGHT
        out_degree_score = user_data['Out_Degree'] * OUT_DEGREE_WEIGHT
        return k_shell_score + out_degree_score

    def calculate_user_feature_score(self, user_data):
        """
        Calculate the feature-based score for a user.

        Parameters:
            user_data (dict): Dictionary containing user information.

        Returns:
            float: The calculated user feature score.
        """
        score = sum(user_data[feature] * weight for feature, weight in zip(FEATURES, FEATURE_WEIGHTS))
        return score

    def calculate_total_score(self, topological_score, user_feature_score, website_impressions, bonus=DEFAULT_BONUS):
        """
        Calculate the total score for a user.

        Parameters:
            topological_score (float): The topological score of the user.
            user_feature_score (float): The feature-based score of the user.
            website_impressions (float): The calculated website impressions.
            bonus (float, optional): Bonus score to be added. Defaults to DEFAULT_BONUS.

        Returns:
            float: The calculated total score.
        """
        return BETA * topological_score + GAMMA * user_feature_score + ALPHA * website_impressions + (1 - ALPHA - BETA - GAMMA) * bonus

    def get_user_scores(self, all_user_data):
        """
        Update the user data with calculated scores for all users.

        Parameters:
            all_user_data (dict): Dictionary containing information for all users.
                                 Keys are user IDs and values are dictionaries containing user details.

        Returns:
            dict: Updated user data with calculated scores for each user.
        """
        updated_user_data = {}

        for user_id, user_data in all_user_data.items():
            # Replace None values with defaults from DEFAULT_VALUES
            user_data = {key: (value if value is not None else DEFAULT_VALUES.get(key, 0)) for key, value in user_data.items()}

            # Calculate individual scores
            topological_score = self.calculate_topological_score(user_data)
            user_feature_score = self.calculate_user_feature_score(user_data)
            website_impressions = user_data['Unique_Pageviews'] * (1 + user_data['Scroll_Depth_Percent'] / 100)
            
            # Scale and round individual scores
            topological_score = round(topological_score * 100)
            user_feature_score = round(user_feature_score * 100)
            website_impressions = round(website_impressions * 100)

            total_score = self.calculate_total_score(
                topological_score / 100, user_feature_score / 100, website_impressions / 100, bonus=user_data['Bonus']
            )
            total_score = round(total_score * 100)

            # Add calculated scores to the user data
            updated_user_data[user_id] = {
                'Topological_Score': topological_score,
                'User_Feature_Score': user_feature_score,
                'Website_Impressions': website_impressions,
                'Bonus': user_data['Bonus'],
                'Posts_Created_ts': user_data['Posts_created_ts'],
                'Total_Score': total_score,
            }

        return updated_user_data