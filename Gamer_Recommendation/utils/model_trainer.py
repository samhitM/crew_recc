import tensorflow as tf
from datetime import datetime
import os

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001, batch_size=128, epochs=65):
        # Initialize the trainer with model, learning rate, batch size, and epochs
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def compile_model(self):
        # Compile the model with Adam optimizer and MSE loss
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='mean_squared_error')

    def train_model(self, train_dataset, test_dataset):
        # Compile the model and train it on the given datasets
        self.compile_model()
        history = self.model.fit(train_dataset, validation_data=test_dataset, epochs=self.epochs)
        
         # Save model after training
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        save_path = f'models/siamese_model_{timestamp}.keras'
        os.makedirs('models', exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")
        return history  # Return the training history
