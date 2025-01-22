DATABASE_TABLE = "crew_user"

# Weights for gaming activity
HOUR_WEIGHT = 0.5  # Weight for maximum gaming hours
ACHIEVEMENT_WEIGHT = 0.5  # Weight for achievements

# Composite scoring weights
ALPHA = 0.4  # Weight for gaming activity score
BETA = 0.4  # Weight for total impression score
GAMMA = 0.1  # Weight for bonus factors (engagement, contributions, etc.)

# Social interaction weights
DELTA = 0.1  # Weight for social interactions
EPSILON = 0.05  # Weight for longevity
ETA = 0.05  # Weight for cross-platform engagement


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