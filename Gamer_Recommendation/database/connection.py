import psycopg2
from fastapi import HTTPException
import core.config as config

def get_db_connection(database_name=None):
    """Establishes a connection to the specified PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=database_name or config.DATABASE,  # Default to crewdb
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        return connection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to {database_name or config.DATABASE}: {str(e)}")
