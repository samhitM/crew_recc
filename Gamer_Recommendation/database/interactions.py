from database.connection import get_db_connection
from fastapi import HTTPException
from database.queries import fetch_from_db

def get_interaction_type(user_id, player_ids, database_name="crewdb"):
    """Fetches the latest 'friend_request' and 'ignored' interactions separately for each player_id."""
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

    interaction_map = {}

    for _, entity_id, interaction_type, action, metadata, create_ts in results:
        if entity_id not in interaction_map:
            interaction_map[entity_id] = {}

        # Store the latest "ignored" and "friend_request" separately
        if action == "ignored" and "ignored" not in interaction_map[entity_id] and interaction_type == 'SWIPE':
            interaction_map[entity_id]["ignored"] = {
                "interactionType": interaction_type,
                "metadata": metadata,
                "createTimestamp": create_ts,
            }

        elif action == "friend_request" and "friend_request" not in interaction_map[entity_id] and (interaction_type == 'SWIPE' or interaction_type == 'PROFILE_INTERACTION'):
            interaction_map[entity_id]["friend_request"] = {
                "interactionType": interaction_type,
                "metadata": metadata,
                "createTimestamp": create_ts,
            }

    return interaction_map