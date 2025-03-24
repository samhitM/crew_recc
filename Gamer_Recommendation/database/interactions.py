from database.connection import get_db_connection
from fastapi import HTTPException
from database.queries import fetch_from_db

def get_interaction_type(user_id, player_ids, database_name="crewdb"):
    """Batch fetches interactions for a list of player_ids."""
    if not player_ids:
        return {}

    query = """
        SELECT user_id, entity_id_primary, interaction_type, action, metadata, create_ts
        FROM user_interactions
        WHERE (user_id = %s AND entity_id_primary = ANY(%s) AND entity_primary = 'USER')
           OR (entity_id_primary = %s AND user_id = ANY(%s) AND entity_primary = 'USER')
        ORDER BY create_ts DESC;
    """
    results = fetch_from_db(query, (user_id, player_ids, user_id, player_ids), database_name)

    interaction_map = {
        entity_id: {
            "interactionType": interaction_type,
            "action": action,
            "metadata": metadata,
            "createTimestamp": create_ts,
        }
        for _, entity_id, interaction_type, action, metadata, create_ts in results
    }
    return interaction_map

        
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