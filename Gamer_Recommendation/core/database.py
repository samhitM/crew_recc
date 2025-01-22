import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2
from psycopg2 import sql
from fastapi import HTTPException
import core.config as config
from levelScoring.constants import DATABASE_TABLE
from typing import Dict, List

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
        
def get_interaction_type(user_a_id: str, user_b_id: str):
    """
    Identifies the type of interaction between two users based on the updated `user_interactions` table.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT interaction_type, action, metadata, create_ts
                FROM user_interactions
                WHERE (user_id = %s AND entity_id_primary = %s AND entity_primary = 'USER')
                   OR (user_id = %s AND entity_id_primary = %s AND entity_primary = 'USER')
                ORDER BY create_ts DESC
                LIMIT 1;
            """
            cur.execute(query, (user_a_id, user_b_id, user_b_id, user_a_id))
            result = cur.fetchone()

            if result:
                return {
                    "interactionType": result[0],
                    "action": result[1],
                    "metadata": result[2],
                    "createTimestamp": result[3],
                }
            else:
                return {}  # Return an empty dictionary if no interaction is found
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching interaction type: {str(e)}")
    finally:
        conn.close()
        
def get_specialisations(user_id: str):
    """
    Fetches the specialisations of a user from the `crew_user` table.
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
        
def get_user_message_stats(user_id: str):
    """
    Fetches total messages grouped by topic ID for a given user,
    along with topic members and last message time, sorted by message count and topic members' length.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    tm.topic_id,
                    tm.member_id,
                    tm.role,
                    tm.is_muted,
                    COUNT(m.id) AS total_messages,
                    ARRAY_AGG(DISTINCT tm.member_id) AS topic_members,
                    MAX(m.created_ts) AS last_message_time
                FROM 
                    topic_member tm
                LEFT JOIN 
                    message m ON tm.topic_id = m.topic_id
                WHERE 
                    tm.member_id = %s
                GROUP BY 
                    tm.topic_id, tm.member_id, tm.role, tm.is_muted
                ORDER BY 
                    total_messages DESC, LENGTH(ARRAY_TO_STRING(ARRAY_AGG(DISTINCT tm.member_id), ',')) DESC;
            """
            cur.execute(query, (user_id,))
            results = cur.fetchall()

            # Process results into a dictionary format for JSON-like response
            data = [
                {
                    "topicId": row[0],
                    "memberId": row[1],
                    "role": row[2],
                    "isMuted": row[3],
                    "totalMessages": row[4],
                    "topicMembers": row[5],
                    "lastMessageTime": row[6],
                }
                for row in results
            ]
            return {"userStats": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user message stats: {str(e)}")
    finally:
        conn.close()

def fetch_db_user_value(
    id_value: str,
    column: str,
    table: str,
    id_field: str = "user_id",
    default_value=0,
    limit: int = 1,
    conditions: List[Dict[str, str]] = None
):
    """
    Fetch a value from the database with enhanced flexibility and error handling.

    Args:
        id_value (str): The value of the identifier field.
        column (str): The column to fetch.
        table (str): The table to query.
        id_field (str): The identifier field name.
        default_value: Value to return if no result is found or in case of an error.
        limit (int): Maximum number of rows to fetch. Defaults to 1.
        conditions (List[Dict[str, str]]): Additional conditions for the query in the form of
                                           [{"field": "field_name", "operator": "=", "value": "value"}].

    Returns:
        Any: The value from the specified column, or `default_value` if no value is found or an error occurs.
    """
    conn = get_db_connection()
    try:
        # Build base query
        query_parts = [
            sql.SQL("SELECT {column} FROM {table} WHERE {id_field} = %s").format(
                column=sql.Identifier(column),
                table=sql.Identifier(table),
                id_field=sql.Identifier(id_field)
            )
        ]

        # Append additional conditions
        query_params = [id_value]
        if conditions:
            for condition in conditions:
                query_parts.append(
                    sql.SQL("AND {field} {operator} %s").format(
                        field=sql.Identifier(condition["field"]),
                        operator=sql.SQL(condition["operator"])
                    )
                )
                query_params.append(condition["value"])

        # Add LIMIT clause
        query_parts.append(sql.SQL("LIMIT %s"))
        query_params.append(limit)

        # Combine query parts
        query = sql.SQL(" ").join(query_parts)

        # Execute query
        with conn.cursor() as cur:
            cur.execute(query, query_params)
            results = cur.fetchmany(limit)
            if not results:
                return default_value
            return results if limit > 1 else results[0][0]

    except Exception as e:
        print("Here",id_value)
        print(f"Database query error: {str(e)}")
        return default_value
    finally:
        conn.close()


def update_crew_level(user_id: str, level: int, score: float):
    """
    Updates the crew level and composite score for a user in the database.

    Args:
        user_id (str): The unique identifier of the user.
        level (int): The new crew level to be assigned to the user.
        score (float): The composite score to associate with the crew level.

    Raises:
        HTTPException: If an error occurs while updating the database.
    """
    # Add field for level_score(float) and impression_score(float) in the crew_user table 
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = f"""
                UPDATE {DATABASE_TABLE} 
                SET crew_level = %s
                WHERE user_id = %s;
            """
            cur.execute(query, (level, user_id))
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating crew level for user {user_id}: {str(e)}")
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
        

def update_crew_impressions(user_id: str, impression_score: float):
    """
    Updates the crew impression for a user in the database.

    Args:
        user_id (str): The unique identifier of the user.
        impression_score (float): The new crew impression score to be assigned to the user.

    Raises:
        HTTPException: If an error occurs while updating the database.
    """
    # Add field for impression_score(float) in the crew_user table 
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = f"""
                UPDATE {DATABASE_TABLE} 
                SET crew_impression = %s
                WHERE user_id = %s;
            """
            cur.execute(query, (impression_score, user_id))
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating crew impression for user {user_id}: {str(e)}")
    finally:
        conn.close()

        


