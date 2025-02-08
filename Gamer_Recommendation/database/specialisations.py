from fastapi import HTTPException
from database.connection import get_db_connection

def get_specialisations(user_id: str):
    """
    Fetches the specialisations of a user from the `crew_user` table.

    Parameters:
    - user_id (str): The unique identifier of the user whose specialisations need to be fetched.

    Returns:
    - list: A list of specialisations if found; otherwise, an empty list.

    Raises:
    - HTTPException (500): If an error occurs while querying the database.
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
