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
        
def get_user_message_stats(user_id: str):
    """
    Fetches total messages grouped by topic ID for a given user,
    along with topic members and last message time, sorted by message count and topic members' length.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    tm.topic_id,
                    tm.member_id,
                    tm.role,
                    tm.is_muted,
                    COUNT(m.id) AS total_messages,
                    ARRAY_AGG(DISTINCT tm.member_id) AS topic_members,
                    MAX(m.created_ts) AS last_message_time
                FROM 
                    topic_member tm
                LEFT JOIN 
                    message m ON tm.topic_id = m.topic_id
                WHERE 
                    tm.member_id = %s
                GROUP BY 
                    tm.topic_id, tm.member_id, tm.role, tm.is_muted
                ORDER BY 
                    total_messages DESC, LENGTH(ARRAY_TO_STRING(ARRAY_AGG(DISTINCT tm.member_id), ',')) DESC;
            """
            cur.execute(query, (user_id,))
            results = cur.fetchall()

            # Process results into a dictionary format for JSON-like response
            data = [
                {
                    "topicId": row[0],
                    "memberId": row[1],
                    "role": row[2],
                    "isMuted": row[3],
                    "totalMessages": row[4],
                    "topicMembers": row[5],
                    "lastMessageTime": row[6],
                }
                for row in results
            ]
            return {"userStats": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user message stats: {str(e)}")
    finally:
        conn.close()


def fetch_field(user_id: str, column: str, table: str, default_value=0):
    """
    Fetches a specific field for a user from the given table.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = sql.SQL("SELECT {column} FROM {table} WHERE user_id = %s").format(
                column=sql.Identifier(column),
                table=sql.Identifier(table)
            )
            cur.execute(query, (user_id,))
            result = cur.fetchone()
            return result[0] if result and result[0] is not None else default_value
    except Exception as e:
        return default_value
    finally:
        conn.close()

def update_crew_level(user_id: str, level: int, score: float):
    """
    Updates the crew level and composite score for a user in the database.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                UPDATE user_levels
                SET crew_level = %s, composite_score = %s
                WHERE user_id = %s;
            """
            cur.execute(query, (level, score, user_id))
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating crew level for user {user_id}: {str(e)}")
    finally:
        conn.close()
        
if __name__ == "__main__":
    # Test the new function
    user_id = "4iGFY2u1rrh" 
    try:
        result = get_user_message_stats(user_id)
        print(result)  
    except Exception as e:
        print(f"Error during function testing: {e}")

        


