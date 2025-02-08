from database.connection import get_db_connection
from psycopg2 import sql
from typing import List, Dict, Optional, Any

def fetch_all_users_data(
    table: str,
    database_name: str,
    columns: Optional[List[str]] = None,
    conditions: Optional[List[Dict[str, str]]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    group_by: Optional[List[str]] = None 
) -> List[Dict[str, Any]]:
    """
    Fetch multiple user records from the database with optional filters.
    Supports aliasing, aggregation, and grouping.

    Args:
        table (str): The table name.
        database_name (str): The database to query.
        columns (List[str], optional): List of columns to fetch, including aliases or SQL expressions.
        conditions (List[Dict[str, str]], optional): Filtering conditions [{field, operator, value}].
        order_by (str, optional): Column to order results by.
        limit (int, optional): Maximum number of records to fetch.
        group_by (List[str], optional): List of columns to group by (needed for aggregate functions).

    Returns:
        List[Dict[str, Any]]: List of records as dictionaries.
    """
    conn = get_db_connection(database_name)
    try:
        with conn.cursor() as cur:
            # Validate and format columns (allow aliases and expressions)
            if columns:
                valid_columns = []
                for col in columns:
                    if " AS " in col or "(" in col:  # Detect SQL expressions (COUNT, SUM, etc.)
                        valid_columns.append(sql.SQL(col))  # Keep raw SQL
                    else:
                        valid_columns.append(sql.Identifier(col))  # Safe identifier
            else:
                valid_columns = [sql.SQL("*")]  # Default to all columns

            # Prepare query parts
            query_parts = [
                sql.SQL("SELECT {} FROM {}").format(
                    sql.SQL(", ").join(valid_columns), sql.Identifier(table)
                )
            ]

            query_params = []
            where_clauses = []

            # Apply conditions
            if conditions:
                for condition in conditions:
                    where_clauses.append(
                        sql.SQL("{} {} %s").format(
                            sql.Identifier(condition["field"]),
                            sql.SQL(condition["operator"])
                        )
                    )
                    query_params.append(condition["value"])

            if where_clauses:
                query_parts.append(sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_clauses))

            # Add GROUP BY clause if needed
            if group_by:
                query_parts.append(sql.SQL("GROUP BY {}").format(
                    sql.SQL(", ").join(map(sql.Identifier, group_by))
                ))

            # Order results
            if order_by:
                query_parts.append(sql.SQL("ORDER BY {}").format(sql.Identifier(order_by)))

            # Limit results
            if limit:
                query_parts.append(sql.SQL("LIMIT %s"))
                query_params.append(limit)

            # Final query
            query = sql.SQL(" ").join(query_parts)
            cur.execute(query, query_params)

            # Get column names
            result_columns = [desc[0] for desc in cur.description]
            results = cur.fetchall()

            # Convert results to dictionaries
            final_results = [dict(zip(result_columns, row)) for row in results]

            return final_results

    except Exception as e:
        print(f"Database query error in {database_name}: {str(e)}")
        return []
    finally:
        conn.close()

# def fetch_all_users_data(
#     table: str,
#     database_name: str,
#     columns: Optional[List[str]] = None,
#     conditions: Optional[List[Dict[str, str]]] = None,
#     order_by: Optional[str] = None,
#     limit: Optional[int] = None,
#     aggregate: str = None,  # Supports COUNT, SUM
#     single_value: bool = False,  # Indicates single value fetch
#     default_value: Any = None
# ) -> Any:
#     """
#     Fetch user records from the database with optional filters.
#     Supports aggregate functions (COUNT, SUM) and single value fetches.
    
#     Args:
#         table (str): The table name.
#         database_name (str): The database to query.
#         columns (List[str], optional): List of columns to fetch. Defaults to all (*).
#         conditions (List[Dict[str, str]], optional): Filtering conditions [{field, operator, value}].
#         order_by (str, optional): Column to order results by.
#         limit (int, optional): Maximum number of records to fetch.
#         aggregate (str, optional): Aggregate function (COUNT, SUM) to apply.
#         single_value (bool, optional): Whether to fetch a single value. Default is False.
#         default_value (Any, optional): Default value if no results found.
    
#     Returns:
#         Any: Aggregate value or list of records.
#     """
#     conn = get_db_connection(database_name)
#     try:
#         with conn.cursor() as cur:
#             # Check available columns in the table
#             cur.execute(
#                 sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s"),
#                 [table]
#             )
#             available_columns = {row[0] for row in cur.fetchall()}

#             # Validate requested columns
#             if columns:
#                 valid_columns = [col for col in columns if col in available_columns]
#                 missing_columns = [col for col in columns if col not in available_columns]
#             else:
#                 valid_columns = ["*"]
#                 missing_columns = []

#             # Handle aggregate functions (COUNT, SUM)
#             if aggregate:
#                 selected_columns = sql.SQL(f"{aggregate}({valid_columns[0]})")
#             else:
#                 selected_columns = sql.SQL(", ").join(map(sql.Identifier, valid_columns)) if valid_columns != ["*"] else sql.SQL("*")

#             # Prepare query
#             query_parts = [
#                 sql.SQL("SELECT {columns} FROM {table}").format(
#                     columns=selected_columns, table=sql.Identifier(table)
#                 )
#             ]

#             query_params = []
#             where_clauses = []

#             # Apply conditions
#             if conditions:
#                 for condition in conditions:
#                     where_clauses.append(
#                         sql.SQL("{field} {operator} %s").format(
#                             field=sql.Identifier(condition["field"]),
#                             operator=sql.SQL(condition["operator"])
#                         )
#                     )
#                     query_params.append(condition["value"])

#             if where_clauses:
#                 query_parts.append(sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_clauses))

#             # Order results
#             if order_by and order_by in available_columns:
#                 query_parts.append(sql.SQL("ORDER BY ") + sql.Identifier(order_by))

#             # Limit results for non-aggregate queries
#             if limit and not aggregate:
#                 query_parts.append(sql.SQL("LIMIT %s"))
#                 query_params.append(limit)

#             query = sql.SQL(" ").join(query_parts)
#             # Execute query
#             cur.execute(query, query_params)
#             results = cur.fetchall()

#             # If fetching single value or aggregate, return first result or default
#             if single_value or aggregate:
#                 return results[0][0] if results else default_value

#             # Process results for non-aggregate queries
#             result_columns = [desc[0] for desc in cur.description]  # Retrieved column names
#             final_results = []
#             for row in results:
#                 row_dict = dict(zip(result_columns, row))
#                 # Assign None to missing columns
#                 for col in missing_columns:
#                     row_dict[col] = None
#                 final_results.append(row_dict)

#             return final_results

#     except Exception as e:
#         print(f"Database query error in {database_name}: {str(e)}")
#         return default_value if single_value or aggregate else []
#     finally:
#         conn.close()


# def fetch_db_user_value(
#     column: str,
#     table: str,
#     database_name: str,
#     id_value: Optional[str] = None,
#     id_field: str = "user_id",
#     conditions: List[Dict[str, str]] = None,
#     default_value=0,
#     aggregate: str = None,  # Supports COUNT, SUM
#     limit: int = 1,
# ):
#     """
#     Fetch a value from the specified database.

#     Args:
#         column (str): The column to fetch (can be COUNT(*), SUM(), etc.).
#         table (str): The table to query.
#         database_name (str): The database to connect to (crewdb or productdb).
#         id_value (str, optional): The value of the identifier field (if applicable).
#         id_field (str): The identifier field name (defaults to "user_id").
#         conditions (List[Dict[str, str]], optional): Additional conditions.
#         default_value: Value to return if no result is found or an error occurs.
#         aggregate (str, optional): Aggregate function (e.g., "COUNT", "SUM").
#         limit (int): Maximum number of rows to fetch (default: 1).

#     Returns:
#         Any: The value from the specified column or `default_value` if not found.
#     """
#     conn = get_db_connection(database_name)
#     try:
#         # Build query dynamically
#         query_parts = [sql.SQL("SELECT")]

#         # Handle aggregate functions (COUNT, SUM)
#         if aggregate:
#             query_parts.append(sql.SQL(f"{aggregate}({column})"))
#         else:
#             query_parts.append(sql.Identifier(column))

#         query_parts.append(sql.SQL("FROM {table}").format(table=sql.Identifier(table)))

#         # WHERE conditions
#         query_params = []
#         where_clauses = []

#         if id_value:
#             where_clauses.append(
#                 sql.SQL("{id_field} = %s").format(id_field=sql.Identifier(id_field))
#             )
#             query_params.append(id_value)

#         if conditions:
#             for condition in conditions:
#                 where_clauses.append(
#                     sql.SQL("{field} {operator} %s").format(
#                         field=sql.Identifier(condition["field"]),
#                         operator=sql.SQL(condition["operator"])
#                     )
#                 )
#                 query_params.append(condition["value"])

#         # Add WHERE clause if conditions exist
#         if where_clauses:
#             query_parts.append(sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_clauses))

#         # Add LIMIT clause
#         if not aggregate:
#             query_parts.append(sql.SQL("LIMIT %s"))
#             query_params.append(limit)

#         # Combine query parts
#         query = sql.SQL(" ").join(query_parts)

#         # Execute query
#         with conn.cursor() as cur:
#             cur.execute(query, query_params)
#             results = cur.fetchone()
#             return results[0] if results else default_value

#     except Exception as e:
#         print(f"Database query error in {database_name}: {str(e)}")
#         return default_value
#     finally:
#         conn.close()

# def fetch_all_users_data(
#     table: str,
#     database_name: str,
#     columns: Optional[List[str]] = None,
#     conditions: Optional[List[Dict[str, str]]] = None,
#     order_by: Optional[str] = None,
#     limit: Optional[int] = None
# ) -> List[Dict[str, Any]]:
#     """
#     Fetch multiple user records from the database with optional filters.
#     If a column does not exist, it is mapped to None instead of failing entirely.

#     Args:
#         table (str): The table name.
#         database_name (str): The database to query.
#         columns (List[str], optional): List of columns to fetch. Defaults to all (*).
#         conditions (List[Dict[str, str]], optional): Filtering conditions [{field, operator, value}].
#         order_by (str, optional): Column to order results by.
#         limit (int, optional): Maximum number of records to fetch.

#     Returns:
#         List[Dict[str, Any]]: List of records as dictionaries.
#     """
#     conn = get_db_connection(database_name)
#     try:
#         with conn.cursor() as cur:
#             # Check available columns in the table
#             cur.execute(
#                 sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s"),
#                 [table]
#             )
#             available_columns = {row[0] for row in cur.fetchall()}

#             # Validate requested columns
#             if columns:
#                 valid_columns = [col for col in columns if col in available_columns]
#                 missing_columns = [col for col in columns if col not in available_columns]
#             else:
#                 valid_columns = ["*"]
#                 missing_columns = []
            
#             # Prepare query parts
#             selected_columns = sql.SQL(", ").join(map(sql.Identifier, valid_columns)) if valid_columns != ["*"] else sql.SQL("*")
#             query_parts = [
#                 sql.SQL("SELECT {columns} FROM {table}").format(
#                     columns=selected_columns, table=sql.Identifier(table)
#                 )
#             ]

#             query_params = []
#             where_clauses = []

#             # Apply conditions
#             if conditions:
#                 for condition in conditions:
#                     where_clauses.append(
#                         sql.SQL("{field} {operator} %s").format(
#                             field=sql.Identifier(condition["field"]),
#                             operator=sql.SQL(condition["operator"])
#                         )
#                     )
#                     query_params.append(condition["value"])

#             if where_clauses:
#                 query_parts.append(sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_clauses))

#             # Order results
#             if order_by and order_by in available_columns:
#                 query_parts.append(sql.SQL("ORDER BY ") + sql.Identifier(order_by))

#             # Limit results
#             if limit:
#                 query_parts.append(sql.SQL("LIMIT %s"))
#                 query_params.append(limit)

#             query = sql.SQL(" ").join(query_parts)
#             # Execute query
#             cur.execute(query, query_params)
#             result_columns = [desc[0] for desc in cur.description]  # Retrieved column names
#             results = cur.fetchall()

#             # Process results, ensuring missing columns are mapped to None
#             final_results = []
#             for row in results:
#                 row_dict = dict(zip(result_columns, row))
#                 # Assign None to missing columns
#                 for col in missing_columns:
#                     row_dict[col] = None
#                 final_results.append(row_dict)

#             return final_results

#     except Exception as e:
#         print(f"Database query error in {database_name}: {str(e)}")
#         return []
#     finally:
#         conn.close()

# def fetch_product_data(product_id: str):
#     """Fetches product details from productdb based on product_id."""
#     query = """
#     SELECT product_id, name, currency_price, stock_quantity, status, product_type, product_category
#     FROM product
#     WHERE product_id = %s
#     """
#     try:
#         conn = get_db_connection("productdb")  # Connect to productdb
#         with conn.cursor() as cursor:
#             cursor.execute(query, (product_id,))
#             product_data = cursor.fetchone()
        
#         conn.close()

#         if product_data:
#             print("Product Data:")
#             print(product_data)
#             return {
#                 "product_id": product_data[0],
#                 "name": product_data[1],
#                 "currency_price": product_data[2],  # JSONB field
#                 "stock_quantity": product_data[3],
#                 "status": product_data[4],
#                 "product_type": product_data[5],
#                 "product_category": product_data[6],
#             }
#         return None
#     except Exception as e:
#         print(f"Error fetching product data: {str(e)}")
#         return None

# def fetch_cross_platform_engagement(user_ids: List[str]) -> Dict[str, int]:
#     """
#     Fetches the number of unique platforms each user has connected to in bulk.

#     Args:
#         user_ids (List[str]): List of user IDs.

#     Returns:
#         Dict[str, int]: A dictionary mapping user IDs to their count of unique platforms.
#     """
#     database_name = "crewdb"  # Ensure we query the correct database
#     conn = get_db_connection(database_name)
#     engagement_map = {user_id: 0 for user_id in user_ids}  # Default values

#     if not user_ids:
#         return engagement_map  # Return early if no users provided

#     try:
#         query = """
#             SELECT user_id, COUNT(DISTINCT source_name) AS platform_count
#             FROM user_connections
#             WHERE user_id IN %s
#             GROUP BY user_id;
#         """
#         with conn.cursor() as cur:
#             cur.execute(query, (tuple(user_ids),))
#             results = cur.fetchall()

#             # Update engagement map with actual values
#             for user_id, platform_count in results:
#                 engagement_map[user_id] = platform_count

#     except Exception as e:
#         print(f"Error fetching Cross_Platform_Engagement: {str(e)}. Defaulting to 0 for all users.")
#     finally:
#         conn.close()

#     return engagement_map

