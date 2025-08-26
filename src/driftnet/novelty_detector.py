import torch
import torch.nn as nn
import numpy as np
from .autoencoder import AutoEncoder

class RegimeNoveltyDetector:
    """
    Detects novel market regimes based on the reconstruction error of an autoencoder.
    """
    def __init__(self, input_dim: int, latent_dim: int, threshold: float = 1.5, window_size: int = 100):
        """
        Initializes the novelty detector.

        Args:
            input_dim (int): The input dimension for the autoencoder.
            latent_dim (int): The latent dimension for the autoencoder.
            threshold (float): The factor by which the current error must exceed the
                               running average to be considered novel (e.g., 1.5 = 50% higher).
            window_size (int): The number of recent samples to use for the running average.
        """
        self.autoencoder = AutoEncoder(input_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

        self.threshold = threshold
        self.window_size = window_size
        self.error_history = []
        self.running_avg_error = 0.0

    def train_autoencoder(self, data_loader: torch.utils.data.DataLoader, epochs: int = 5):
        """
        Trains the internal autoencoder on a dataset.

        Args:
            data_loader: A PyTorch DataLoader providing the training data.
            epochs (int): The number of epochs to train for.
        """
        print("Training novelty detector's autoencoder...")
        self.autoencoder.train()
        for epoch in range(epochs):
            for data_batch in data_loader:
                # Assuming data_loader yields batches of sequences
                x = data_batch[0] if isinstance(data_batch, list) else data_batch
                x = x.view(x.size(0), -1) # Flatten the sequence

                self.optimizer.zero_grad()
                reconstructed = self.autoencoder(x)
                loss = self.criterion(reconstructed, x)
                loss.backward()
                self.optimizer.step()
        print("Autoencoder training complete.")

    def check_novelty(self, x: torch.Tensor) -> bool:
        """
        Checks if the input sample `x` represents a novelty.

        Args:
            x (torch.Tensor): The input sample tensor (a single sequence).

        Returns:
            bool: True if the sample is considered novel, False otherwise.
        """
        x = x.view(1, -1) # Ensure it's a batch of 1
        current_error = self.autoencoder.get_reconstruction_error(x).item()

        is_novel = False
        # Only start detecting after the initial window is filled
        if len(self.error_history) > self.window_size:
            if current_error > self.running_avg_error * self.threshold:
                is_novel = True

        # Update the running average
        self.error_history.append(current_error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)

        if len(self.error_history) > 0:
            self.running_avg_error = np.mean(self.error_history)

        return is_novel
