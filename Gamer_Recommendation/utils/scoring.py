from datetime import datetime, timezone
import numpy as np
import pandas as pd

class ScoreAdjuster:
    @staticmethod
    def adjust_scores(df, interaction_map, relationships):
        """
        Adjusts the recommendation score based on user interactions and relationships.

        Parameters:
            df (pd.DataFrame): A DataFrame containing user data, including 'player_id' and 'score'.
            interaction_map (dict): Maps user IDs to their interactions, each containing multiple actions.
            relationships (dict): Dictionary containing sets of related users (e.g., "friends").

        Returns:
            pd.DataFrame: The DataFrame with adjusted scores.
        """
        scores = df['score'].values
        current_time = datetime.now(timezone.utc)  # Ensure current_time is timezone-aware
        interactions = df['player_id'].map(interaction_map).apply(lambda x: x if isinstance(x, dict) else {})

        # Masks for friend_request with correct interaction type
        friend_request_mask = interactions.apply(
            lambda x: "friend_request" in x and x["friend_request"].get("interactionType") in {"SWIPE", "PROFILE_INTERACTION"}
        )

        # Masks for ignored with correct interaction type
        ignored_mask = interactions.apply(
            lambda x: "ignored" in x and x["ignored"].get("interactionType") == "SWIPE"
        )

        # Apply friend request penalty (0.5x score)
        scores[friend_request_mask.to_numpy()] *= 0.5

        # Apply decay factor for ignored users
        ignored_indices = np.where(ignored_mask.to_numpy())[0]
        if len(ignored_indices) > 0:
            time_elapsed = []
            for i in ignored_indices:
                ignored_data = interactions.iloc[i].get("ignored", {})
                timestamp_str = ignored_data.get("createTimestamp")
                
                if isinstance(timestamp_str, datetime):
                    timestamp = timestamp_str.replace(tzinfo=timezone.utc)
                    time_elapsed.append((current_time - timestamp).days)
                else:
                    time_elapsed.append(30)  # Default max decay if timestamp is missing
            
            time_elapsed = np.array(time_elapsed)

            # Identify users ignored in the last 7 days
            ignore_filter = time_elapsed < 7
            ignored_recently = df.iloc[ignored_indices[ignore_filter]]['player_id'].tolist()
            decay_factors = np.maximum(0, 1 - time_elapsed / 30)
            scores[ignored_indices] *= decay_factors
            
            df['score'] = scores
            # Remove users present in ignored_recently
            df = df[~df['player_id'].isin(ignored_recently)]

        # Friend-based adjustments
        friend_mask = df['player_id'].isin(relationships.get("friends", set())).to_numpy()
        df.loc[friend_mask, 'score'] *= 0.8 

        return df