from fastapi import HTTPException
from database.connection import get_db_connection
from psycopg2 import sql

def get_endpoint_data():
    """
    Fetches endpoint data and associated user IDs from the `source_endpoint` table.

    Returns:
    - dict: A dictionary containing a list of player statistics under the key `playersStats`. 
            Each entry includes:
            - userId (str): The unique identifier of the user.
            - endpoint_data (dict): A nested dictionary with:
                - endpointData (Any): The raw data retrieved from the `endpoint_data` column.

    Raises:
    - HTTPException (404): If no data is found for the specified endpoint.
    - HTTPException (500): If a database error occurs during execution.
    """

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
