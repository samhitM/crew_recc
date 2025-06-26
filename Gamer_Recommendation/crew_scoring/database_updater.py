import sys
import os
import pandas as pd
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import get_db_connection, release_db_connection

class CrewScoringUpdater:
    """
    Updates the database with revised crew impression and level scores.
    """
    
    def __init__(self):
        self.database_name = "crewdb"
    
    def update_crew_impressions(self, impression_data: List[Dict]):
        """
        Update crew impression scores in the database.
        
        Args:
            impression_data: List of dictionaries with user_id and impression scores
        """
        if not impression_data:
            print("No impression data to update")
            return
        
        conn = get_db_connection(self.database_name)
        try:
            with conn.cursor() as cur:
                # Update users table with impression scores
                for record in impression_data:
                    user_id = record['user_id']
                    impression_score = record['total_impression_score']
                    pagerank = record.get('pagerank', 0)
                    
                    # Update or insert impression score
                    update_query = """
                    UPDATE users 
                    SET crew_impression = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """
                    
                    cur.execute(update_query, (impression_score, user_id))
                
                conn.commit()
                print(f"✅ Updated impression scores for {len(impression_data)} users")
                
        except Exception as e:
            conn.rollback()
            print(f"❌ Error updating impression scores: {e}")
        finally:
            release_db_connection(conn, self.database_name)
    
    def update_crew_levels(self, level_data: List[Dict]):
        """
        Update crew levels in the database.
        
        Args:
            level_data: List of dictionaries with user_id and crew levels
        """
        if not level_data:
            print("No level data to update")
            return
        
        conn = get_db_connection(self.database_name)
        try:
            with conn.cursor() as cur:
                # Update users table with crew levels
                for record in level_data:
                    user_id = record['user_id']
                    crew_level = record['crew_level']
                    composite_score = record.get('composite_score', 0)
                    
                    # Update or insert crew level
                    update_query = """
                    UPDATE users 
                    SET crew_level = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """
                    
                    cur.execute(update_query, (crew_level, user_id))
                
                conn.commit()
                print(f"✅ Updated crew levels for {len(level_data)} users")
                
        except Exception as e:
            conn.rollback()
            print(f"❌ Error updating crew levels: {e}")
        finally:
            release_db_connection(conn, self.database_name)
    
    def update_from_csv_files(self, impression_file: str = None, level_file: str = None):
        """
        Update database from CSV files.
        
        Args:
            impression_file: Path to impression scores CSV file
            level_file: Path to crew levels CSV file
        """
        # Update impressions if file provided
        if impression_file and os.path.exists(impression_file):
            try:
                impression_df = pd.read_csv(impression_file)
                impression_data = impression_df.to_dict('records')
                self.update_crew_impressions(impression_data)
            except Exception as e:
                print(f"❌ Error reading impression file {impression_file}: {e}")
        
        # Update levels if file provided
        if level_file and os.path.exists(level_file):
            try:
                level_df = pd.read_csv(level_file)
                level_data = level_df.to_dict('records')
                self.update_crew_levels(level_data)
            except Exception as e:
                print(f"❌ Error reading level file {level_file}: {e}")

def main():
    """Main function to update database with latest scores."""
    print("UPDATING DATABASE WITH REVISED CREW SCORES")
    print("="*50)
    
    updater = CrewScoringUpdater()
    
    # Define file paths
    impression_file = "crew_impressions_revised.csv"
    level_file = "crew_levels_revised.csv"
    
    # Update from CSV files
    updater.update_from_csv_files(impression_file, level_file)
    
    print("Database update completed!")

if __name__ == "__main__":
    main()
