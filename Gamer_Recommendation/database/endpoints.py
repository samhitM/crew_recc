from fastapi import HTTPException
from database.connection import get_db_connection
from psycopg2 import sql
from database.queries import fetch_from_db

def get_endpoint_data(database_name="crewdb"):
    """
    Fetches endpoint data and associated user IDs from the `source_endpoint` table.
    
    Returns:
    - dict: Contains a list of player statistics under `playersStats`:
        - userId (str)
        - endpoint_data (dict) -> endpointData (Any)
    
    Raises:
    - HTTPException (404): If no data is found.
    - HTTPException (500): If a database error occurs.
    """

    query = """
        SELECT endpoint_data, user_id
        FROM source_endpoint
        WHERE endpoint = %s
    """
    results = fetch_from_db(query, ('getCompletePlayerData',), database_name)
    
    players_stats = [{"userId": user_id, "endpoint_data": {"endpointData": data}} for data, user_id in results]
    
    if not players_stats:
        raise HTTPException(status_code=404, detail="No data found for the specified endpoint")
    
    return {"playersStats": players_stats}
