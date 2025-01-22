from impressionScoring.constants import K_SHELL_WEIGHT, OUT_DEGREE_WEIGHT, FEATURE_WEIGHTS, BETA, GAMMA, DEFAULT_BONUS, DEFAULT_VALUES

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
        features = [
            'Reposts', 'Replies', 'Mentions', 'Favorites', "Total_Messages", 'Interest_Topic',
            'Bio_Content', 'Profile_Likes', 'User_Games', 'Verified_Status', 'Posts_on_Topic'
        ]
        score = sum(user_data[feature] * weight for feature, weight in zip(features, FEATURE_WEIGHTS))
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
        return BETA * topological_score + GAMMA * user_feature_score + (1 - BETA - GAMMA) * website_impressions + bonus

    def update_user_scores(self, user_data):
        """
        Update the user data with calculated scores.

        Parameters:
            user_data (dict): Dictionary containing user information. Missing values are replaced with defaults.

        Returns:
            dict: Updated user data with calculated scores.
        """
        # Replace None values with defaults from DEFAULT_VALUES
        user_data = {key: (value if value is not None else DEFAULT_VALUES.get(key, 0)) for key, value in user_data.items()}
        
        topological_score = self.calculate_topological_score(user_data)
        user_feature_score = self.calculate_user_feature_score(user_data)
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
