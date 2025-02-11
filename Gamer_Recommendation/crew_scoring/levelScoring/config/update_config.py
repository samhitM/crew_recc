from typing import Dict, List

class UpdateConfig:
    """
    This class holds constants for default values and update configuration used across the application.
    """

    # Table names, database names, and key columns for each entity type
    DATABASES = {
        "crew_user": {
            "table": "crew_user",
            "database_name": "crewdb",
            "key_column": "user_id"
        }
    }

    @staticmethod
    def get_update_for_table(table_key: str) -> Dict[str, object]:
        """
        Returns update configuration for performing bulk updates based on the table key and update data.

        Parameters:
        - table_key (str): The key identifying the table (e.g., 'user_tiers', 'product').

        Returns:
        - Dict[str, object]: A dictionary with the update configuration (table, database name, updates, key column).
        """
        # Fetch the table, database, and key column configuration
        table_config = UpdateConfig.DATABASES.get(table_key, {})

        # Return the constructed update configuration
        return {
            "table": table_config.get("table"),
            "database_name": table_config.get("database_name"),
            "key_column": table_config.get("key_column")
        }