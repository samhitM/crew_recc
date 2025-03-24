from datetime import datetime
import numpy as np
import pandas as pd

class ScoreAdjuster:
    @staticmethod
    def adjust_scores(df, interaction_map, relationships):
        """
        Adjusts the recommendation score based on user interactions and relationships.

        Parameters:
            df (pd.DataFrame): A DataFrame containing user data, including 'player_id' and 'score'.
            interaction_map (dict): Maps user IDs to their last interaction details.
            relationships (dict): Dictionary containing sets of related users (e.g., "friends").

        Returns:
            pd.DataFrame: The DataFrame with adjusted scores.
        """
        scores = df['score'].values
        # Interaction-based adjustments
        interactions = df['player_id'].map(interaction_map).apply(lambda x: x if isinstance(x, dict) else {})  # Ensure valid dict
        profile_interaction_mask = interactions.apply(lambda x: x.get('interactionType') == "PROFILE_INTERACTION")
        friend_request_mask = profile_interaction_mask & interactions.apply(lambda x: x.get('action') == "friend_request")
        ignored_mask = profile_interaction_mask & interactions.apply(lambda x: x.get('action') == "ignored")
    
        # Apply friend request penalty
        scores[friend_request_mask] *= 0.5

        # Apply decay factor for ignored users
        ignored_indices = ignored_mask[ignored_mask].index
        if len(ignored_indices) > 0:
            time_elapsed = np.array([
                (datetime.now() - interaction_map[player_id]['createTimestamp']).days
                if 'createTimestamp' in interaction_map.get(player_id, {}) else 30
                for player_id in df.loc[ignored_indices, 'player_id']
            ])
            decay_factors = np.maximum(0, 1 - time_elapsed / 30)
            scores[ignored_indices] *= decay_factors

        # Friend-based adjustments
        friend_mask = df['player_id'].isin(relationships.get("friends", set()))
        scores[friend_mask] *= 0.8

        df['score'] = scores
        return df