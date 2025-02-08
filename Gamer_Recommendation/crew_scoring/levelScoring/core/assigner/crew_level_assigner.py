from typing import List, Dict
from database import fetch_all_users_data
from crew_scoring.levelScoring.config import QueryConfig
    
class CrewLevelAssigner:
    def __init__(self):
        """
        A utility class for assigning crew levels to users based on their scores and predefined thresholds.
        """
        self.previous_levels = self.fetch_previous_levels()
    
    def fetch_previous_levels(self) -> Dict[int, int]:
        """
        Fetches the previous crew levels of users from the database.

        Returns:
            Dict[int, int]: A dictionary mapping user IDs to their respective crew levels.
                            Only users with a non-null crew level are included.
        """
        query_config = QueryConfig.get_query_for_table("crew_levels", [])
        user_data = fetch_all_users_data(
            table=query_config['table'],
            database_name=query_config['database_name'],
            columns=query_config['columns']
        )
        return {record["user_id"]: int(record["crew_level"]) for record in user_data if record["crew_level"] is not None}
    
    
    def assign_crew_level(self, user_ids: List[int], scores: List[float], thresholds: List[float]) -> List[int]:
        """
        Assigns a crew level to each score based on the given thresholds.

        Parameters:
            user_ids (List[int]): A list of users for which levels need to be assigned.
            scores (List[float]): A list of scores for which levels need to be assigned.
            thresholds (List[float]): A list of thresholds that define level boundaries.
                                       The thresholds must be sorted in ascending order.

        Returns:
            List[int]: A list of levels corresponding to each score. Levels are assigned
                       as integers starting from 1 (lowest level).
        """
        levels = []
        for user_id, score in zip(user_ids, scores):
            previous_level = self.previous_levels.get(user_id, 1)
            new_level = previous_level  # Default to previous level
            
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    new_level = max(previous_level, min(i + 1, previous_level + 1))  # Ensure linear progression
                    break
            else:
                new_level = max(previous_level, min(len(thresholds) + 1, previous_level + 1))  # Ensure max +1 increment
                
            self.previous_levels[user_id] = new_level
            levels.append(new_level)
        
        return levels

