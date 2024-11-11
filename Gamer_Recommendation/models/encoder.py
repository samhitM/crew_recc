from tensorflow.keras import layers, Model

class Encoder(Model):
    def __init__(self, input_dim, embedding_dim):
        super(Encoder, self).__init__()
        # First dense layer with 128 units and ReLU activation
        self.dense1 = layers.Dense(128, activation='relu')
        # Second dense layer that outputs an embedding of specified dimension with ReLU activation
        self.dense2 = layers.Dense(embedding_dim, activation='relu')

    def call(self, inputs):
        # Apply the first dense layer to the inputs
        x = self.dense1(inputs)
        # Pass the result through the second dense layer to obtain the embedding
        x = self.dense2(x)
        return x
