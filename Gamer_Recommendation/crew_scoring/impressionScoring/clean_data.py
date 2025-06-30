"""
Clean impression data by removing rows with empty user_ids.
"""
import pandas as pd

def clean_impression_data():
    """Clean impression data and save cleaned version."""
    try:
        # Read the current impression data
        df = pd.read_csv("crew_impressions_revised.csv")
        print(f"Original data: {len(df)} rows")
        
        # Filter out rows with empty user_ids
        df_clean = df[df['user_id'].notna() & (df['user_id'] != '')]
        print(f"Cleaned data: {len(df_clean)} rows")
        print(f"Removed {len(df) - len(df_clean)} rows with empty user_ids")
        
        # Save cleaned data
        df_clean.to_csv("crew_impressions_revised.csv", index=False)
        print("Cleaned impression data saved successfully")
        
        # Show sample of user_ids
        print(f"Sample user_ids: {df_clean['user_id'].head(10).tolist()}")
        
    except Exception as e:
        print(f"Error cleaning impression data: {e}")

if __name__ == "__main__":
    clean_impression_data()
