import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2
from psycopg2 import sql
from fastapi import HTTPException
import core.config as config

def get_db_connection():
    """Attempts to connect to the PostgreSQL database using configuration settings."""
    try:
        connection = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DATABASE,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        return connection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to the database: {str(e)}")

def get_endpoint_data():
    """Fetches endpoint data and user IDs from the `source_endpoint` table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = sql.SQL("""
                SELECT endpoint_data, user_id
                FROM source_endpoint
                WHERE endpoint = %s
            """)
            cur.execute(query, ('getCompletePlayerData',))
            results = cur.fetchall()
            if results:
                players_stats = [{"userId": row[1], "endpoint_data": {"endpointData": row[0]}} for row in results]
                return {"playersStats": players_stats}
            else:
                raise HTTPException(status_code=404, detail="No data found for the specified endpoint")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    finally:
        conn.close()

def get_username(user_id):
    """Retrieves the full name of a user by their user ID from the `user` table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT full_name FROM \"user\" WHERE id = %s;", (user_id,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching username: {str(e)}")
    finally:
        conn.close()
        
def get_interaction_type(user_a_id: str, user_b_id: str):
    """
    Identifies the type of interaction between two users based on the `user_interactions` table.
    """
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT event_type, action, metadata, create_date, create_ts
                FROM user_interactions
                WHERE user_id = %s AND interacted_user_id = %s
                   OR user_id = %s AND interacted_user_id = %s
                ORDER BY create_ts DESC
                LIMIT 1;
            """
            cur.execute(query, (user_a_id, user_b_id, user_b_id, user_a_id))
            result = cur.fetchone()

            if result:
                return {
                    "eventType": result[0],
                    "action": result[1],
                    "metadata": result[2],
                    "createDate": result[3],
                    "createTimestamp": result[4],
                }
            else:
                return {}  # Return an empty dictionary if no interaction is found
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching interaction type: {str(e)}")
    finally:
        conn.close()
        
def get_specialisations(user_id: str):
    """
    Fetches the specialisations of a user from the `crew_user` table.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT specialisation
                FROM crew_user
                WHERE user_id = %s;
            """
            cur.execute(query, (user_id,))
            result = cur.fetchone()

            if result and result[0]:
                return result[0]  # Return the array of specialisations
            else:
                return []  # Return an empty list if no specialisations are found
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching specialisations: {str(e)}")
    finally:
        conn.close()

