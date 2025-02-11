from levelScoring.config.update_config import UpdateConfig
from database import perform_table_updates
from typing import List, Dict, Any

class CrewLevelUpdater:
    """
    This class is responsible for updating crew levels in the crew_user table.
    It fetches the necessary update configurations and performs the updates.
    """

    def __init__(self):
        """
        Initializes the CrewLevelUpdater.
        """
        # Load the update configuration for the crew_user table
        self.update_config = UpdateConfig.get_update_for_table(table_key="crew_user")
    
    def update_crew_levels(self, updates: List[Dict[str, Any]]) -> None:
        """
        Updates the crew levels for a list of users in the crew_user table.

        Parameters:
        - updates (List[Dict[str, Any]]): List of dictionaries containing 'user_id' and 'crew_level'.
        """
        # Perform the update operation
        perform_table_updates(
            table=self.update_config["table"],
            database_name=self.update_config["database_name"],
            updates=updates,
            key_column=self.update_config["key_column"]
        )