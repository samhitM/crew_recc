"""
Database operations for level scoring.
"""
import os
import pandas as pd
from typing import Dict, List
import psycopg2


class LevelDatabaseManager:
    """Manages all database operations for level scoring."""
    
    def __init__(self):
        self.db_config = {
            "host": "34.44.52.84",
            "port": 5432,
            "user": "admin_crew",
            "password": "xV/nI2+=uOI&KL1P",
            "database": "crewdb"
        }
    
    def get_db_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def fetch_user_games_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch gaming data from user_games table."""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cur:
                query = "SELECT user_id, gaming_time FROM user_games LIMIT 10000"
                cur.execute(query)
                results = cur.fetchall()
                
                gaming_data = {}
                for row in results:
                    user_id = row[0]
                    gaming_time = float(row[1] or 0)
                    
                    gaming_data[user_id] = {
                        'max_hours': gaming_time,
                        'avg_hours': gaming_time,  # Use same value for both
                        'days_active': 1 if gaming_time > 0 else 0
                    }
                
                print(f"Fetched gaming data for {len(gaming_data)} users")
                return gaming_data
                
        except Exception as e:
            print(f"Error fetching user games data: {e}")
            return {}
        finally:
            conn.close()
    
    def fetch_friendship_data(self) -> List[Dict]:
        """Fetch friendship relations from the database."""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cur:
                query = """
                SELECT user_a_id, user_b_id, relation 
                FROM friendship 
                WHERE state = true
                LIMIT 10000
                """
                cur.execute(query)
                results = cur.fetchall()
                
                friendship_data = []
                for row in results:
                    friendship_data.append({
                        'user_a_id': row[0],
                        'user_b_id': row[1],
                        'relation': row[2]
                    })
                
                print(f"Fetched {len(friendship_data)} friendship records for community detection")
                return friendship_data
                
        except Exception as e:
            print(f"Error fetching friendship data: {e}")
            return []
        finally:
            conn.close()
    
    def fetch_user_interactions(self) -> List[Dict]:
        """Fetch user interactions from the database."""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cur:
                query = """
                SELECT user_id, entity_id_primary, interaction_type, action 
                FROM user_interactions 
                LIMIT 50000
                """
                cur.execute(query)
                results = cur.fetchall()
                
                interaction_data = []
                for row in results:
                    interaction_data.append({
                        'user_id': row[0],
                        'entity_id_primary': row[1],
                        'interaction_type': row[2],
                        'action': row[3]
                    })
                
                print(f"Fetched {len(interaction_data)} user interaction records for community detection")
                return interaction_data
                
        except Exception as e:
            print(f"Error fetching user interactions data: {e}")
            return []
        finally:
            conn.close()
    
    def get_impression_scores(self) -> Dict[str, float]:
        """Get normalized impression scores from the CSV file created by impression calculator."""
        print("Getting normalized impression scores...")
        
        try:
            # Look for the impression scores file in the impressionScoring folder
            impression_file = "../impressionScoring/crew_impressions_revised.csv"
            if os.path.exists(impression_file):
                df = pd.read_csv(impression_file)
                # Filter out rows with empty user_ids
                df = df[df['user_id'].notna() & (df['user_id'] != '')]
                
                # Use normalized total impression scores for proper normalization in level scoring
                impression_scores = {}
                for _, row in df.iterrows():
                    impression_scores[row['user_id']] = row['norm_total_impression_score']
                
                print(f"Loaded normalized impression scores for {len(impression_scores)} users from {impression_file}")
                return impression_scores
            else:
                # Try alternative locations
                alt_paths = [
                    "crew_impressions_revised.csv",  # Current directory
                    "../../crew_impressions_revised.csv",  # Parent directory
                    "../crew_impressions_revised.csv"  # One level up
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        df = pd.read_csv(alt_path)
                        # Filter out rows with empty user_ids
                        df = df[df['user_id'].notna() & (df['user_id'] != '')]
                        
                        impression_scores = {}
                        for _, row in df.iterrows():
                            impression_scores[row['user_id']] = row['norm_total_impression_score']
                        print(f"Loaded normalized impression scores for {len(impression_scores)} users from {alt_path}")
                        return impression_scores
                
                print("No impression scores file found in any expected location, using default values")
                return {}
                
        except Exception as e:
            print(f"Error getting impression scores: {e}")
            return {}
