import psycopg2
from fastapi import HTTPException
from psycopg2.pool import SimpleConnectionPool
import core.config as config

# def get_db_connection(database_name=None):
#     """Establishes a connection to the specified PostgreSQL database."""
#     try:
#         connection = psycopg2.connect(
#             host=config.DB_HOST,
#             port=config.DB_PORT,
#             database=database_name or config.DATABASE,  # Default to crewdb
#             user=config.DB_USER,
#             password=config.DB_PASSWORD
#         )
#         return connection
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error connecting to {database_name or config.DATABASE}: {str(e)}")

# Connection pools per database
db_pools = {}

def init_db_pools():
    """Initialize connection pools for configured databases."""
    global db_pools
    for db_name, db_config in config.DB_CONFIG.items(): 
        db_pools[db_name] = SimpleConnectionPool(
            minconn=1, maxconn=10, 
            host=db_config["host"], port=db_config["port"], 
            user=db_config["user"], password=db_config["password"], 
            database=db_name
        )

def get_db_connection(database_name="crewdb"):
    """Fetches a connection from the connection pool."""
    init_db_pools()
    if database_name not in db_pools:
        raise HTTPException(status_code=500, detail=f"Database {database_name} not configured")

    try:
        return db_pools[database_name].getconn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to {database_name}: {str(e)}")
    
def release_db_connection(conn, database_name="crewdb"):
    """Releases the connection back to the pool."""
    if database_name in db_pools and conn:
        db_pools[database_name].putconn(conn)
