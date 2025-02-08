from typing import Dict, List

class QueryConfig:
    """
    This class holds constants for default values and query configuration used across the application.
    """
    
    # Table names, database names, and columns for each entity type
    DATABASES = {
        "user_tiers": {
            "table": "user",
            "database_name": "crewdb",
            "columns": ["id", "user_tiers"]
        },
        "posts_replies": {
            "table": "posts",
            "database_name": "crewdb",
            "columns": ["parent_id", "COUNT(*) AS \"Replies\""],
            "group_by": ["parent_id"],
            "conditions": [{"field": "post_status", "operator": "=", "value": "Published"}]
        },
        "posts_mentions": {
            "table": "posts",
            "database_name": "crewdb",
            "columns": ["user_id", "COUNT(*) AS \"Mentions\""],
            "group_by": ["user_id"]
        },
        "posts_favorites": {
            "table": "posts",
            "database_name": "crewdb",
            "columns": ["user_id", "SUM((reaction_emojis->>'favorite')::int) AS \"Favorites\""],
            "group_by": ["user_id"]
        },
        "posts_created_ts": {  
            "table": "posts",
            "database_name": "crewdb",
            "columns": ["user_id", "created_ts"],  
            "conditions": [
                {"field": "post_status", "operator": "=", "value": "Published"}  
            ],
            "order_by": "created_ts"  
        }
    }

    @staticmethod
    def get_query_for_table(table_key: str, entity_ids: List[str]) -> Dict[str, object]:
        # Fetch the table, database, and columns configuration
        table_config = QueryConfig.DATABASES.get(table_key, {})
        
        # Define conditions for the query, such as filtering by entity IDs
        conditions = table_config.get("conditions", [])
        if entity_ids:
            if table_key in ["user_tiers", "posts_mentions", "posts_favorites", "total_messages", "posts_created_ts"]:
                condition_field = "user_id" if table_key != "user_tiers" else "id"
                conditions.append({"field": condition_field, "operator": "IN", "value": tuple(entity_ids)})
        
        return {
            "table": table_config.get("table"),
            "database_name": table_config.get("database_name"),
            "columns": table_config.get("columns"),
            "conditions": conditions,
            "group_by": table_config.get("group_by", []),
            "order_by": table_config.get("order_by", None)
        }