from datetime import datetime

class ScoreAdjuster:
    @staticmethod
    def adjust_score(row, interaction_map, relationships):
        """
        Adjusts the recommendation score based on user interactions and relationships.

        Parameters:
            row (pd.Series): A row of the DataFrame containing user data, including 'player_id' and 'score'.
            interaction_map (dict): Maps user IDs to their last interaction details.
            relationships (dict): Dictionary containing sets of related users (e.g., "friends").

        Returns:
            float: The adjusted score.
        """
    
        score = row['score']
        interaction = interaction_map.get(row['player_id'])

        if interaction:
            if interaction['interactionType'] == "PROFILE_INTERACTION" and interaction['action'] == "friend_request":
                score *= 0.5
            elif interaction['interactionType'] == "PROFILE_INTERACTION" and interaction['action'] == "ignored":
                time_elapsed = (datetime.now() - interaction['createTimestamp']).days
                decay_factor = max(0, 1 - time_elapsed / 30)
                score *= decay_factor

        # Adjust score for friends
        if row['player_id'] in relationships.get("friends", []):
            score *= 0.8
        
        return score