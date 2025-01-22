from typing import List

# Crew Level Assigner
class CrewLevelAssigner:
    """
    A utility class for assigning crew levels to users based on their scores and predefined thresholds.
    """

    @staticmethod
    def assign_crew_level(scores: List[float], thresholds: List[float]) -> List[int]:
        """
        Assigns a crew level to each score based on the given thresholds.

        Each score is compared against the thresholds in order. The assigned level
        corresponds to the first threshold that the score does not exceed. If the score
        exceeds all thresholds, it is assigned the highest level.

        Parameters:
            scores (List[float]): A list of scores for which levels need to be assigned.
            thresholds (List[float]): A list of thresholds that define level boundaries.
                                       The thresholds must be sorted in ascending order.

        Returns:
            List[int]: A list of levels corresponding to each score. Levels are assigned
                       as integers starting from 1 (lowest level).
        """
        levels = []
        for score in scores:
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    levels.append(i + 1)
                    break
            else:
                levels.append(len(thresholds) + 1)  # If no threshold is higher
        return levels
