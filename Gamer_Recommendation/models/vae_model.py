import tensorflow as tf
from tensorflow.keras import layers, Model

class VAE(Model):
    def __init__(self, num_users, num_games, embedding_dim):
        # Initialize the Variational Autoencoder (VAE) model
        super(VAE, self).__init__()
        
        # Number of unique users and games in the dataset, embedding dimension for latent space
        self.num_users = num_users
        self.num_games = num_games
        self.embedding_dim = embedding_dim

        # Encoder part: Fully connected layer to map user input to a latent space
        self.encoder = layers.Dense(128, activation='relu')
        
        # Latent space parameters: Mean (mu) and log variance (log_var)
        self.mu = layers.Dense(embedding_dim)  # Mean of the latent distribution
        self.log_var = layers.Dense(embedding_dim)  # Log-variance of the latent distribution

        # Decoder part: Fully connected layer to reconstruct the input from the latent space
        self.decoder = layers.Dense(128, activation='relu')
        
        # Output layer: Reconstruct the original input, using sigmoid activation for probability output
        self.output_layer = layers.Dense(num_games, activation='sigmoid')

    def call(self, user_input):
        # Forward pass for VAE: Encoding -> Sampling from latent space -> Decoding
        x = self.encoder(user_input)  # Encode the user input
        mu = self.mu(x)  # Get the mean of the latent distribution
        log_var = self.log_var(x)  # Get the log variance of the latent distribution
        
        # Reparameterization trick: Sample from the latent space using mu and log_var
        epsilon = tf.random.normal(shape=tf.shape(mu))  # Sample epsilon from a normal distribution
        z = mu + tf.exp(0.5 * log_var) * epsilon  # Reparameterization trick to get the latent variable z
        
        # Decode the latent variable z back into the original input space
        x = self.decoder(z)  # Decode the latent variable
        reconstructed = self.output_layer(x)  # Output the reconstructed game scores

        # Return the reconstructed output and the parameters of the latent distribution (mu, log_var)
        return reconstructed, mu, log_var

    def vae_loss(self, original, reconstructed, mu, log_var):
        # VAE loss consists of two parts: reconstruction loss and KL divergence loss

        # Reconstruction loss: Mean squared error between the original and reconstructed inputs
        reconstruction_loss = tf.reduce_mean(tf.square(original - reconstructed))
        
        # KL divergence loss: Measures how much the learned latent distribution diverges from a standard normal distribution
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        
        # Total VAE loss: Sum of reconstruction loss and KL divergence loss
        return reconstruction_loss + kl_loss
