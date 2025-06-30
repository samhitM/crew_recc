"""
Verify that level scoring only contains users from impression scoring.
"""
import pandas as pd

def verify_user_consistency():
    """Verify user consistency between impression and level scoring."""
    try:
        # Read impression data
        impression_df = pd.read_csv("../impressionScoring/crew_impressions_revised.csv")
        impression_users = set(impression_df[impression_df['user_id'].notna() & (impression_df['user_id'] != '')]['user_id'])
        
        # Read level data
        level_df = pd.read_csv("crew_levels_revised.csv")
        level_users = set(level_df[level_df['user_id'].notna() & (level_df['user_id'] != '')]['user_id'])
        
        print(f"Impression users: {len(impression_users)}")
        print(f"Level users: {len(level_users)}")
        
        # Check if all level users are in impression users
        level_not_in_impression = level_users - impression_users
        impression_not_in_level = impression_users - level_users
        
        print(f"Level users not in impression: {len(level_not_in_impression)}")
        if level_not_in_impression:
            print(f"Users: {list(level_not_in_impression)[:10]}")
        
        print(f"Impression users not in level: {len(impression_not_in_level)}")
        if impression_not_in_level:
            print(f"Users: {list(impression_not_in_level)[:10]}")
        
        # Check for empty user_ids in level data
        empty_level_users = level_df[level_df['user_id'].isna() | (level_df['user_id'] == '')]
        print(f"Empty user_ids in level data: {len(empty_level_users)}")
        
        if len(level_not_in_impression) == 0 and len(empty_level_users) == 0:
            print("SUCCESS: All level users are present in impression data and no empty user_ids!")
        else:
            print("WARNING: There are inconsistencies in the data")
            
    except Exception as e:
        print(f"Error verifying user consistency: {e}")

if __name__ == "__main__":
    verify_user_consistency()
