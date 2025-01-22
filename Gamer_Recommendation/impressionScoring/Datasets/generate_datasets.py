import pandas as pd
import numpy as np
import random

# Function to generate synthetic dataset
def generate_dataset(num_users=1000, output_file="crew_impression_dataset.csv"):
    np.random.seed(42)
    random.seed(42)
    
   
    data = {
        'User_ID': [f"User_{i+1}" for i in range(num_users)],
        'K_Shell': np.random.uniform(1, 5, num_users),
        'Out_Degree': np.random.randint(1, 50, num_users),
        'Reposts': np.random.randint(0, 100, num_users),
        'Replies': np.random.randint(0, 50, num_users),
        'Mentions': np.random.randint(0, 30, num_users),
        'Favorites': np.random.randint(0, 200, num_users),
        'Interest_Topic': np.random.uniform(0, 1, num_users),
        'Bio_Content': np.random.randint(0, 2, num_users),  # Binary: 0 or 1
        'Profile_Likes': np.random.randint(0, 500, num_users),
        'User_Games': np.random.randint(1, 20, num_users),
        'Verified_Status': np.random.choice([0, 1], num_users),  # Binary: 0 or 1
        'Posts_on_Topic': np.random.randint(0, 100, num_users),
        'Bonus': np.random.uniform(0, 1, num_users),
        'Unique_Pageviews': np.random.randint(10, 500, num_users),
        'Scroll_Depth_Percent': np.random.uniform(10, 100, num_users),
        'Timestamp': pd.date_range("2023-01-01", periods=num_users, freq="H"),
        'Impressions': np.random.randint(100, 1000, num_users),
        'Engagement_Per_Impression': np.random.uniform(0.1, 1.0, num_users)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Synthetic dataset saved to {output_file}")

# Generate dataset
generate_dataset()
