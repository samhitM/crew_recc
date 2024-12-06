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
