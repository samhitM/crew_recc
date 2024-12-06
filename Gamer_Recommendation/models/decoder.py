from tensorflow.keras import layers, Model
import tensorflow as tf

class Decoder(Model):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        # First dense layer with ReLU activation for non-linearity
        self.dense1 = layers.Dense(128, activation='relu')
        # Second dense layer to produce a single output score for each user-game pair
        self.dense2 = layers.Dense(1)  # Final output shape: (batch_size, 1)

    def call(self, user_embedding, game_embedding):
        # Concatenate user and game embeddings along the last axis
        combined = tf.concat([user_embedding, game_embedding], axis=-1)
        # Pass the concatenated embeddings through the first dense layer
        x = self.dense1(combined)
        # Pass the result through the second dense layer to get a final score
        x = self.dense2(x)  # Shape: (batch_size, 1)
        return x
