import numpy as np
from minisom import MiniSom

class SimpleSOM:
    """
    A wrapper for the MiniSom library to train a Self-Organizing Map.
    """
    def __init__(self, grid_size=(10, 10), input_len=3, sigma=1.0, learning_rate=0.5):
        """
        Initializes the SimpleSOM.
        """
        if not isinstance(grid_size, tuple) or len(grid_size) != 2:
            raise ValueError("grid_size must be a tuple of (width, height)")

        self.grid_size = grid_size
        self.som = MiniSom(grid_size[0], grid_size[1], input_len,
                           sigma=sigma, learning_rate=learning_rate)

    def train(self, data, num_iteration=100):
        """
        Trains the SOM on the provided data.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")

        self.som.train_random(data, num_iteration)
        return self

    def get_quantization_error(self, data):
        """
        Calculates the quantization error of the trained SOM.
        """
        return self.som.quantization_error(data)

    def winner(self, vector):
        """
        Finds the winning neuron for a given input vector.
        """
        return self.som.winner(vector)
