# Weight parameters for topological score calculation
K_SHELL_WEIGHT = 0.6
OUT_DEGREE_WEIGHT = 0.4

# Weights for user feature score calculation
FEATURE_WEIGHTS = [0.25, 0.15, 0.2, 0.1, 0.02, 0.2, 0.05, 0.05, 0.03, 0.05]

FEATURES = [
    'Reposts', 'Replies', 'Mentions', 'Favorites', "Total_Messages", 'Interest_Topic',
    'Bio_Content', 'User_Games', 'Verified_Status', 'Posts_on_Topic'
]

# Missing : Reposts, Interest_Topic, Posts_on_Topic

# Weight parameters for total score calculation
BETA = 0.5
GAMMA = 0.3

# Default bonus value
DEFAULT_BONUS = 0

# Default values for missing fields in user data
DEFAULT_VALUES = {
    'Reposts': 0.0,
    'Replies': 0.0,
    'Mentions': 0.0,
    'Favorites': 0.0,
    'Total_Messages': 0.0,
    'Interest_Topic': 0.0,
    'Bio_Content': 0.0,
    'User_Games': 0.0,
    'Verified_Status': 0.0,
    'Posts_on_Topic': 0.0,
    'K_Shell': 0.0,
    'Out_Degree': 0.0,
    'Unique_Pageviews': 0.0,
    'Scroll_Depth_Percent': 0.0,
    'Bonus': 0.0,
    'Posts_created_ts': None  # Default to None when no posts exist
}
