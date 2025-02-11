import matplotlib.pyplot as plt

class LossPlotter:
    @staticmethod
    def plot_loss(history):
        """Plot the training and validation loss over epochs."""
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()