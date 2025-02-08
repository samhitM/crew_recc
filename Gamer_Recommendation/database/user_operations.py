from database.connection import get_db_connection
from fastapi import HTTPException

def get_username(user_id):
    """
    Retrieves the full name of a user by their user ID from the `user` table.

    Parameters:
    - user_id (str): The unique identifier of the user whose full name is to be retrieved.

    Returns:
    - str or None: The full name of the user if found; otherwise, None.

    Raises:
    - HTTPException (500): If an error occurs while querying the database.
    """

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

def fetch_all_user_ids():
    """
    Fetches all user IDs from the crew_user table.

    Returns:
        list: A list of user IDs (strings) from the crew_user table.

    Raises:
        HTTPException: If an error occurs while fetching user IDs from the database.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = "SELECT user_id FROM crew_user;"
            cur.execute(query)
            result = cur.fetchall()  # Fetch all rows as a list of tuples
            user_ids = [row[0] for row in result]  # Extract user_id from each tuple
            return user_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user IDs: {str(e)}")
    finally:
        conn.close()