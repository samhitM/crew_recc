from core.database import fetch_field, update_crew_level
from typing import Dict

# Constants for thresholds and weights
THRESHOLDS = [50, 100, 200, 300]
WEIGHTS = {
    'hour_weight': 0.5,  # Weight for maximum gaming hours
    'achievement_weight': 0.5,  # Weight for achievements
    'alpha': 0.4,  # Weight for gaming activity score
    'beta': 0.4,  # Weight for total impression score
    'gamma': 0.1,  # Weight for bonus factors (engagement, contributions, etc.)
    'delta': 0.1,  # Weight for social interactions
    'epsilon': 0.05,  # Weight for longevity
    'eta': 0.05  # Weight for cross-platform engagement
}

# Default values for fields
DEFAULT_VALUES = {
    'Max_Hours': 0.0,  # Default maximum gaming hours
    'Achievements': 0.0,  # Default number of achievements
    'Total_Impression_Score': 0.0,  # Default impression score
    'Consistent_Engagement': 0.0,  # Default engagement consistency score
    'Community_Contributions': 0.0,  # Default community contribution score
    'Verified_Status': 0.0,  # Default verification status (not verified)
    'Event_Participation': 0.0,  # Default event participation score
    'Social_Interactions': 0.0,  # Default social interaction score
    'Longevity': 0.0,  # Default longevity score
    'Cross_Platform_Engagement': 0.0  # Default cross-platform engagement score
}

# Data Fetcher Class
class UserDataFetcher:
    """
    Fetches user data from the database.
    """
    @staticmethod
    def fetch_user_data(user_id: str) -> Dict[str, float]:
        """
        Fetch all required fields for a user from the database, falling back to default values if needed.
        """
        fields_to_fetch = {
            'Max_Hours': ('max_hours', 'gaming_activity'),
            'Achievements': ('achievements', 'gaming_activity'),
            'Total_Impression_Score': ('impression_score', 'user_impressions'),
            'Consistent_Engagement': ('engagement', 'user_metrics'),
            'Community_Contributions': ('contributions', 'user_metrics'),
            'Verified_Status': ('verified_status', 'user_metrics'),
            'Event_Participation': ('event_participation', 'user_events'),
            'Social_Interactions': ('social_interactions', 'user_metrics'),
            'Longevity': ('longevity', 'user_lifecycle'),
            'Cross_Platform_Engagement': ('cross_platform', 'user_metrics')
        }
        user_data = {}
        for field, (column, table) in fields_to_fetch.items():
            try:
                # Fetch field from the database
                user_data[field] = fetch_field(user_id, column, table)
            except Exception as e:
                # Use default value if fetching fails
                print(f"Error fetching {field} for User ID {user_id}: {str(e)}. Using default value: {DEFAULT_VALUES[field]}")
                user_data[field] = DEFAULT_VALUES[field]
        return user_data

# Scoring Class
class ScoreCalculator:
    """
    Calculates the composite score for a user.
    """
    @staticmethod
    def calculate_composite_score(data: Dict[str, float]) -> float:
        """
        Computes a weighted composite score based on user data.
        """
        # Gaming activity score (based on max hours and achievements)
        gaming_activity_score = (
            data['Max_Hours'] * WEIGHTS['hour_weight'] +
            data['Achievements'] * WEIGHTS['achievement_weight']
        )
        # Impression score from user interactions
        impression_score = data['Total_Impression_Score']
        # Bonus factors for user engagement, contributions, and event participation
        bonus_factors = (
            data['Consistent_Engagement'] * 0.2 +
            data['Community_Contributions'] * 0.2 +
            data['Verified_Status'] * 0.1 +
            data['Event_Participation'] * 0.5
        )
        # Other contributing factors
        social_interactions_score = data['Social_Interactions'] * WEIGHTS['delta']
        longevity_score = data['Longevity'] * WEIGHTS['epsilon']
        cross_platform_score = data['Cross_Platform_Engagement'] * WEIGHTS['eta']
        
        # Final composite score
        composite_score = (
            WEIGHTS['alpha'] * gaming_activity_score +
            WEIGHTS['beta'] * impression_score +
            WEIGHTS['gamma'] * bonus_factors +
            social_interactions_score +
            longevity_score +
            cross_platform_score
        )
        return composite_score

# Crew Level Assigner
class CrewLevelAssigner:
    """
    Assigns crew levels based on composite score.
    """
    @staticmethod
    def assign_crew_level(score: float) -> int:
        """
        Determines the crew level based on thresholds.
        """
        for i, threshold in enumerate(THRESHOLDS):
            if score < threshold:
                return i + 1
        return len(THRESHOLDS) + 1

# Main Function
def main():
    """
    Main execution function to process users, calculate scores, assign levels, and update the database.
    """
    user_ids = ['4iGFY2u1rrh', '674FY2u1rrh', '3dGFY2u1rrh', 'b5GFY2u1rrh'] 
    for user_id in user_ids:
        try:
            # Fetch user data
            user_data = UserDataFetcher.fetch_user_data(user_id)
            # Calculate composite score
            composite_score = ScoreCalculator.calculate_composite_score(user_data)
            # Assign crew level
            crew_level = CrewLevelAssigner.assign_crew_level(composite_score)
            # Update database with the new level and score
            update_crew_level(user_id, crew_level, composite_score)
            print(f"Updated User ID {user_id}: Composite Score = {composite_score:.2f}, Crew Level = {crew_level}")
        except Exception as e:
            print(f"Failed to process User ID {user_id}: {str(e)}")

if __name__ == "__main__":
    main()
