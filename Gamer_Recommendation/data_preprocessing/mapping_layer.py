class MappingLayer:
    def __init__(self, df):
        # Create mappings from user_id and game_id to integer indices
        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(df['player_id'].unique())}
        self.game_id_mapping = {game_id: idx for idx, game_id in enumerate(df['game_id'].unique())}

        # Create reverse mappings for integer indices back to user_id and game_id
        self.reverse_user_id_mapping = {idx: user_id for user_id, idx in self.user_id_mapping.items()}
        self.reverse_game_id_mapping = {idx: game_id for game_id, idx in self.game_id_mapping.items()}

    def map_user_id(self, user_id):
        # Map a user_id to its corresponding integer index
        return self.user_id_mapping.get(user_id, None)

    def map_game_id(self, game_id):
        # Map a game_id to its corresponding integer index
        return self.game_id_mapping.get(game_id, None)

    def reverse_map_user_id(self, idx):
        # Map an integer index back to its original user_id
        return self.reverse_user_id_mapping.get(idx, None)

    def reverse_map_game_id(self, idx):
        # Map an integer index back to its original game_id
        return self.reverse_game_id_mapping.get(idx, None)
