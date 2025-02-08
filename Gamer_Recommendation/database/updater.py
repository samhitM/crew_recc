from database.connection import get_db_connection
from typing import List, Dict, Any
from psycopg2 import sql

def perform_table_updates(
    table: str,
    database_name: str,
    updates: List[Dict[str, Any]],
    key_column: str
) -> None:
    """
    Executes update queries on a specified table using the provided update data.

    Parameters:
    - table (str): The table name to update.
    - database_name (str): The database to connect to.
    - updates (List[Dict[str, Any]]): A list of dictionaries where each dictionary represents 
                                      a record with columns as keys and their new values.
    - key_column (str): The primary key column used to match records for update.

    Raises:
    - Exception: If an error occurs during the update.
    """
    if not updates:
        return  # No updates to perform

    conn = get_db_connection(database_name)
    try:
        with conn.cursor() as cur:
            # Extract all columns (excluding key_column) that need to be updated
            update_columns = list(updates[0].keys())
            if key_column not in update_columns:
                raise ValueError(f"Key column '{key_column}' must be present in updates.")

            update_columns.remove(key_column)  # Exclude key_column from SET values

            # Build the SQL query dynamically
            query = sql.SQL("""
                UPDATE {table} AS t
                SET {updates}
                FROM (VALUES {values}) AS v ({columns})
                WHERE t.{key_column} = v.{key_column}
            """).format(
                table=sql.Identifier(table),
                updates=sql.SQL(", ").join([
                    sql.SQL("{col} = v.{col}").format(col=sql.Identifier(col)) for col in update_columns
                ]),
                values=sql.SQL(", ").join([
                    sql.SQL("({})").format(sql.SQL(", ").join(sql.Placeholder() * len(update_columns + [key_column])))
                    for _ in updates
                ]),
                columns=sql.SQL(", ").join(map(sql.Identifier, update_columns + [key_column])),
                key_column=sql.Identifier(key_column)
            )

            # Flatten values for placeholders
            values = [tuple(record[col] for col in update_columns + [key_column]) for record in updates]

            # Execute the query
            cur.execute(query, [val for sublist in values for val in sublist])
            conn.commit()

    except Exception as e:
        print(f"Database update error in {database_name}: {str(e)}")
    finally:
        conn.close()