import tensorflow as tf
from tensorflow.keras import Model
from models.matrix_factorization import MatrixFactorization
from models.encoder import Encoder
from models.decoder import Decoder

class SiameseRecommendationModel(Model):
    def __init__(self, num_users, num_games, embedding_dim, **kwargs):
        # Initialize the Siamese Recommendation Model
        super(SiameseRecommendationModel, self).__init__(**kwargs)
        
        # Number of unique users and games in the dataset
        self.num_users = num_users
        self.num_games = num_games
        self.embedding_dim = embedding_dim
        
        # Matrix factorization layer that learns user and game embeddings
        self.matrix_factorization = MatrixFactorization(num_users=num_users, num_games=num_games, embedding_dim=embedding_dim)
        
        # Encoder for game features, mapping to embedding_dim-dimensional vectors
        self.game_encoder = Encoder(input_dim=10, embedding_dim=embedding_dim)
        
        # Encoder for user global features, mapping to embedding_dim-dimensional vectors
        self.user_encoder = Encoder(input_dim=10, embedding_dim=embedding_dim)
        
        # Decoder that combines user and game embeddings and outputs a recommendation score
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def call(self, inputs):
        # Extract the inputs for the model: user and game IDs, game features, and global features
        user_input, game_input, game_features, global_features = inputs
        
        # Get embeddings for users and games from matrix factorization
        user_embedding_mf, game_embedding_mf = self.matrix_factorization(user_input, game_input)
        
        # Encode game-specific features to obtain a dense representation
        game_embedding_encoded = self.game_encoder(game_features)
        
        # Encode user-specific global features to obtain a dense representation
        user_embedding_encoded = self.user_encoder(global_features)
        
        # Combine the matrix factorization embeddings with the encoded embeddings
        final_user_embedding = user_embedding_mf + user_embedding_encoded
        final_game_embedding = game_embedding_mf + game_embedding_encoded
        
        # Pass the final embeddings through the decoder to get the recommendation score
        score = self.decoder(final_user_embedding, final_game_embedding)
        
        # Squeeze the output to remove the extra dimension and return the score
        return tf.squeeze(score, axis=-1)

    def get_config(self):
        # Get configuration parameters for saving the model
        config = super(SiameseRecommendationModel, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_games': self.num_games,
            'embedding_dim': self.embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create an instance of the model from the saved config
        return cls(**config)
