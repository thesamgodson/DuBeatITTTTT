import numpy as np
from minisom import MiniSom

class SimpleSOM:
    """
    A wrapper for the MiniSom library to train a Self-Organizing Map.
    """
    def __init__(self, grid_size=(10, 10), input_len=3, sigma=1.0, learning_rate=0.5):
        """
        Initializes the SimpleSOM.

        Args:
            grid_size (tuple): The (width, height) of the SOM grid.
            input_len (int): The dimension of the input vectors.
            sigma (float): The radius of the different neighbors in the SOM.
            learning_rate (float): The initial learning rate.
        """
        if not isinstance(grid_size, tuple) or len(grid_size) != 2:
            raise ValueError("grid_size must be a tuple of (width, height)")

        self.grid_size = grid_size
        self.som = MiniSom(grid_size[0], grid_size[1], input_len,
                           sigma=sigma, learning_rate=learning_rate)

    def train(self, data, num_iteration=100):
        """
        Trains the SOM on the provided data.

        Args:
            data (np.ndarray): The input data to train the SOM on.
            num_iteration (int): The number of training iterations.

        Returns:
            SimpleSOM: The instance of the trained SOM.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array.")

        self.som.train_random(data, num_iteration)
        return self

    def get_quantization_error(self, data):
        """
        Calculates the quantization error of the trained SOM.

        Args:
            data (np.ndarray): The data to calculate the error against.

        Returns:
            float: The quantization error.
        """
        return self.som.quantization_error(data)

    def winner(self, vector):
        """
        Finds the winning neuron for a given input vector.

        Args:
            vector (np.ndarray): The input vector.

        Returns:
            tuple: The (x, y) coordinates of the winning neuron.
        """
        return self.som.winner(vector)
