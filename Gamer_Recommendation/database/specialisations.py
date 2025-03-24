from fastapi import HTTPException
from database.connection import get_db_connection
from database.queries import fetch_from_db

def get_specialisations(player_ids, database_name="crewdb"):
    """
    Batch fetches specialisations for a list of player_ids.

    Args:
        player_ids (list): List of player IDs to fetch specialisations for.

    Returns:
        dict: A mapping of player IDs to their specialisations.
    """
    if not player_ids:
        return {}
    
    query = """
        SELECT user_id, specialisation
        FROM crew_user
        WHERE user_id = ANY(%s);
    """
    results = fetch_from_db(query, (player_ids,), database_name)

    return {user_id: specialisation if specialisation else [] for user_id, specialisation in results}