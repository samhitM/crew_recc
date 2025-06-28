"""
Database operations for impression scoring.
"""
from typing import Dict, List
import psycopg2


class DatabaseManager:
    """Manages all database operations for impression scoring."""
    
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
                
                print(f"Fetched {len(friendship_data)} friendship records")
                return friendship_data
                
        except Exception as e:
            print(f"Error fetching friendship data: {e}")
            return []
        finally:
            conn.close()
    
    def fetch_user_games_data(self) -> Dict[str, float]:
        """Fetch gaming time data from user_games table."""
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
                    gaming_data[user_id] = gaming_time
                
                print(f"Fetched gaming data for {len(gaming_data)} users")
                return gaming_data
                
        except Exception as e:
            print(f"Error fetching user games data: {e}")
            return {}
        finally:
            conn.close()
    
    def fetch_message_counts(self) -> Dict[str, int]:
        """Fetch message counts from message table based on sender_id."""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cur:
                query = """
                SELECT sender_id, COUNT(*) as message_count 
                FROM message 
                GROUP BY sender_id
                LIMIT 10000
                """
                cur.execute(query)
                results = cur.fetchall()
                
                message_data = {}
                for row in results:
                    sender_id = row[0]
                    message_count = int(row[1] or 0)
                    message_data[sender_id] = message_count
                
                print(f"Fetched message counts for {len(message_data)} users")
                return message_data
                
        except Exception as e:
            print(f"Error fetching message data: {e}")
            return {}
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
                
                print(f"Fetched {len(interaction_data)} user interaction records")
                return interaction_data
                
        except Exception as e:
            print(f"Error fetching user interactions data: {e}")
            return []
        finally:
            conn.close()
