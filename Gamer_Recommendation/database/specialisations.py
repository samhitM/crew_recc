from fastapi import HTTPException
from database.connection import get_db_connection

def get_specialisations(player_ids):
    """
    Batch fetches specialisations for a list of player_ids.

    Args:
        player_ids (list): List of player IDs to fetch specialisations for.

    Returns:
        dict: A mapping of player IDs to their specialisations.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT user_id, specialisation
                FROM crew_user
                WHERE user_id = ANY(%s);
            """
            cur.execute(query, (player_ids,))
            results = cur.fetchall()

            specializations_map = {result[0]: result[1] if result[1] else [] for result in results}
            return specializations_map
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching specialisations: {str(e)}")
    finally:
        conn.close()