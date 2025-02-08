import pandas as pd

class DataFilter:
    @staticmethod
    def process_filters(df, game_id, country=None, recommendation_expertise=None, user_interests=None, age=None, delta=None):
        """
        Filters the DataFrame based on game_id and optional filters (country, expertise, interests, and age range).

        Parameters:
            df (pd.DataFrame): The input dataframe containing user data.
            game_id (int): The game ID to filter on.
            country (str, optional): The country to filter users by.
            recommendation_expertise (str, optional): The expertise to filter users by.
            user_interests (list, optional): List of interests to match.
            age (int, optional): User's age for filtering within a range.
            delta (int, optional): Allowed deviation for age filtering.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        
        # Filter by game_id
        filtered_df = df[df['game_id'] == game_id]
        if filtered_df.empty:
            return filtered_df  # Return early if no matches

        # Filter by country (case-insensitive)
        if country:
            filtered_df = filtered_df[filtered_df['country'].str.lower() == country.lower()]
            if filtered_df.empty:
                return filtered_df

        # Filter by recommendation_expertise (case-insensitive)
        if recommendation_expertise:
            filtered_df = filtered_df[filtered_df['recommendation_expertise'].str.lower() == recommendation_expertise.lower()]
            if filtered_df.empty:
                return filtered_df

        # Filter by user_interests (case-insensitive)
        if user_interests:
            user_interests = [interest.lower() for interest in user_interests]
            filtered_df = filtered_df[
                filtered_df['user_interests'].apply(lambda x: any(interest in [i.lower() for i in x] for interest in user_interests))
            ]
            if filtered_df.empty:
                return filtered_df
            
        # Filter by age range with delta
        if age is not None and delta is not None:
            filtered_df = filtered_df[
                (filtered_df['age'] >= age - delta) & (filtered_df['age'] <= age + delta)
            ]
            if filtered_df.empty:
                return filtered_df

        return filtered_df