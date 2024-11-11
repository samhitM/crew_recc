from tensorflow.keras import layers

class MatrixFactorization(layers.Layer):
    def __init__(self, num_users, num_games, embedding_dim):
        super(MatrixFactorization, self).__init__()
        # Embedding layer for users, mapping each user to an embedding vector
        self.user_factors = layers.Embedding(input_dim=num_users, output_dim=embedding_dim)
        # Embedding layer for games, mapping each game to an embedding vector
        self.game_factors = layers.Embedding(input_dim=num_games, output_dim=embedding_dim)

    def call(self, user_input, game_input):
        # Get the user embedding for each user in the batch
        user_embedding = self.user_factors(user_input)
        # Get the game embedding for each game in the batch
        game_embedding = self.game_factors(game_input)
        return user_embedding, game_embedding
