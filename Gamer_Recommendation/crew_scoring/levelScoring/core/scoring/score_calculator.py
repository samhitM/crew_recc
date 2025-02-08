from levelScoring.config.constants import HOUR_WEIGHT, ACHIEVEMENT_WEIGHT, ALPHA, BETA, GAMMA, DELTA, EPSILON, ETA, ENGAGEMENT_WEIGHT, CONTRIBUTION_WEIGHT, VERIFICATION_WEIGHT, EVENT_PARTICIPATION_WEIGHT, REWARDS_WEIGHT
from typing import List, Dict

# Scoring Class
class ScoreCalculator:
    """
    A utility class for calculating a weighted composite score based on user data.
    """

    @staticmethod
    def calculate_composite_score(data: Dict[str, float]) -> float:
        """
        Computes a weighted composite score based on normalized user data without scaling.

        The composite score is calculated using multiple weighted factors, including
        gaming activity, user impressions, engagement bonuses, social interactions, 
        longevity, and cross-platform engagement.

        Parameters:
            data (Dict[str, float]): A dictionary containing normalized user data. The dictionary
                                     must include the following keys:
                                     - 'Max_Hours': Maximum hours spent gaming.
                                     - 'Achievements': Number of achievements unlocked.
                                     - 'Total_Impression_Score': User's overall impression score.
                                     - 'Consistent_Engagement': A score for consistent activity.
                                     - 'Community_Contributions': Contributions to the community.
                                     - 'Verified_Status': A binary indicator of user verification.
                                     - 'Event_Participation': Engagement in events.
                                     - 'Social_Interactions': A score for social activities.
                                     - 'Longevity': Duration of user activity on the platform.
                                     - 'Cross_Platform_Engagement': Engagement across platforms.

        Returns:
            float: The calculated composite score, which integrates all the weighted factors.
        """
        # Gaming activity score (based on max hours and achievements)
        gaming_activity_score = (
            data['Max_Hours'] * HOUR_WEIGHT +
            data['Achievements'] * ACHIEVEMENT_WEIGHT
        )
        
        # Impression score from user interactions
        impression_score = data['Total_Impression_Score']
        
        # Bonus factors for user engagement, contributions, event participation and rewards
        bonus_factors = (
            data['Consistent_Engagement'] * ENGAGEMENT_WEIGHT +
            data['Community_Contributions'] * CONTRIBUTION_WEIGHT +
            data['Verified_Status'] * VERIFICATION_WEIGHT +
            data['Event_Participation'] * EVENT_PARTICIPATION_WEIGHT +
            data['Rewards'] * REWARDS_WEIGHT
        )
        
        # Other contributing factors
        social_interactions_score = data['Social_Interactions'] * DELTA
        longevity_score = data['Longevity'] * EPSILON
        cross_platform_score = data['Cross_Platform_Engagement'] * ETA

        # Final composite score
        composite_score = (
            ALPHA * gaming_activity_score +
            BETA * impression_score +
            GAMMA * bonus_factors +
            social_interactions_score +
            longevity_score +
            cross_platform_score
        )

        return composite_score  # Can scale the score in future
