from database.connection import get_db_connection
from fastapi import HTTPException

def get_interaction_type(user_a_id: str, user_b_id: str):
    """
    Identifies the type of interaction between two users based on the `user_interactions` table.
    
    Parameters:
    - user_a_id (str): The unique identifier of the first user.
    - user_b_id (str): The unique identifier of the second user.

    Returns:
    - dict: A dictionary containing interaction details if found:
        - interactionType (str): The type of interaction (e.g., 'PROFILE_INTERACTION').
        - action (str): The specific action performed (e.g., 'friend_request', 'ignored').
        - metadata (Any): Additional data stored in the interaction.
        - createTimestamp (datetime): The timestamp of when the interaction occurred.
      If no interaction is found, an empty dictionary `{}` is returned.

    Raises:
    - HTTPException (500): If a database error occurs while fetching interaction data.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT interaction_type, action, metadata, create_ts
                FROM user_interactions
                WHERE (user_id = %s AND entity_id_primary = %s AND entity_primary = 'USER')
                   OR (user_id = %s AND entity_id_primary = %s AND entity_primary = 'USER')
                ORDER BY create_ts DESC
                LIMIT 1;
            """
            cur.execute(query, (user_a_id, user_b_id, user_b_id, user_a_id))
            result = cur.fetchone()

            if result:
                return {
                    "interactionType": result[0],
                    "action": result[1],
                    "metadata": result[2],
                    "createTimestamp": result[3],
                }
            else:
                return {}  # Return an empty dictionary if no interaction is found
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching interaction type: {str(e)}")
    finally:
        conn.close()
        
# def fetch_user_interactions(user_id: str):
#     """
#     Fetches distinct users who have interacted with the given user.
    
#     Parameters:
#         user_id (str): The ID of the user.
    
#     Returns:
#         list: A list of user IDs who interacted with the given user.
#     """
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cur:
#             query = """
#                 SELECT DISTINCT user_id FROM user_interactions
#                 WHERE entity_id_primary = %s AND entity_primary = 'USER'
#             """
#             cur.execute(query, (user_id,))
#             return [row[0] for row in cur.fetchall()]
#     except Exception as e:
#         print(f"Error fetching user interactions for {user_id}: {e}")
#         return []
#     finally:
#         conn.close()