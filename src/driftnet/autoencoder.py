import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    A simple feed-forward autoencoder for dimensionality reduction and novelty detection.
    The reconstruction error (|x - decoder(encoder(x))|) can be used as a novelty score.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initializes the AutoEncoder.

        Args:
            input_dim (int): The dimensionality of the input data (e.g., sequence_length).
            latent_dim (int): The dimensionality of the compressed latent space.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the encoder and decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The reconstructed output tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reconstruction error for a given input batch.

        Args:
            x (torch.Tensor): The input tensor batch.

        Returns:
            torch.Tensor: A tensor containing the reconstruction error for each sample in the batch.
        """
        self.eval() # Ensure the model is in evaluation mode
        with torch.no_grad():
            x_recon = self.forward(x)
            error = torch.mean((x - x_recon)**2, dim=1) # MSE per sample
        return error
