from typing import Dict, List

class QueryConfig:
    """
    This class holds constants for default values and query configuration used across the application.
    """
    
    # Table names, database names, and columns for each entity type
    DATABASES = {
        "crew_user": {
            "table": "crew_user",
            "database_name": "crewdb",
            "columns": ["user_id", "crew_online_time", "crew_badge", "crew_impression", "created_ts", "event_participation", "contributions", "social_interactions"]
        },
        "user_sessions": {
            "table": "user_sessions",
            "database_name": "crewdb",
            "columns": ["user_id", "total_active_time"]
        },
        "user_tiers": {
            "table": "user",
            "database_name": "crewdb",
            "columns": ["id", "user_tiers"]
        },
        "user_product": {
            "table": "user_product",
            "database_name": "productdb",
            "columns": ["user_id", "product_id"]
        },
        "product": {
            "table": "product",
            "database_name": "productdb",
            "columns": ["product_id", "name", "currency_price", "stock_quantity", "status", "product_type", "product_category"]
        },
        "user_connections": {
            "table": "user_connections",
            "database_name": "crewdb",
            "columns": ["user_id", "source_name"]
        },
        "crew_levels": {
            "table": "crew_user",
            "database_name": "crewdb",
            "columns": ["user_id", "crew_level"]
        },
        "user_interactions": {
            "table": "user_interactions",
            "database_name": "crewdb",
            "columns": ["entity_id_primary", "user_id"],
            "conditions": [
                {"field": "entity_id_primary", "operator": "IN", "value": None},  # Placeholder for user_ids
                {"field": "entity_primary", "operator": "=", "value": "USER"}
            ]
        }
    }

    @staticmethod
    def get_query_for_table(table_key: str, entity_ids: List[str]) -> Dict[str, object]:
        """
        Constructs a query configuration for fetching data from the database based on table key and entity IDs.

        Parameters:
        - table_key (str): The key identifying the table (e.g., 'user_tiers', 'product').
        - entity_ids (List[str]): List of IDs to fetch data for (e.g., user IDs or product IDs).

        Returns:
        - Dict[str, object]: A dictionary with the query configuration (table, database name, columns, conditions).
        """
        # Fetch the table, database, and columns configuration
        table_config = QueryConfig.DATABASES.get(table_key, {})
        
        # Define conditions for the query, such as filtering by entity IDs
        conditions = table_config.get("conditions", [])
        if table_key in ["user_product", "user_connections", "crew_user", "user_sessions","crew_levels"]:
            conditions = [{"field": "user_id", "operator": "IN", "value": tuple(entity_ids)}] if entity_ids else []
        elif table_key == "user_tiers":
            conditions = [{"field": "id", "operator": "IN", "value": tuple(entity_ids)}] if entity_ids else []
        elif table_key == "product":
            conditions = [{"field": "product_id", "operator": "IN", "value": tuple(entity_ids)}] if entity_ids else []
        elif table_key == "user_interactions":
                conditions[0]["value"] = tuple(entity_ids)  # Update condition with user_ids

        # Return the constructed query configuration
        return {
            "table": table_config.get("table"),
            "database_name": table_config.get("database_name"),
            "columns": table_config.get("columns"),
            "conditions": conditions
        }