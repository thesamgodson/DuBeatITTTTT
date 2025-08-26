import numpy as np
import torch

class SOMFeatureExtractor:
    """
    Extracts coordinate-based features from a trained SOM for a batch of sequences.
    """
    def __init__(self, som):
        self.som = som

    def extract_features(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Extracts SOM winner coordinates for a batch of price sequences.
        Args:
            sequences (torch.Tensor): A batch of sequences, shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: A batch of SOM features, shape (batch_size, 2).
        """
        device = sequences.device

        # Convert to numpy for processing
        sequences_np = sequences.cpu().numpy()

        batch_features = []
        for seq in sequences_np:
            # Normalize the pattern by its first element to handle relative changes
            normalized_pattern = seq / (seq[0] + 1e-9)

            # Find the winning neuron using the wrapped MiniSom instance
            winner_neuron = self.som.som.winner(normalized_pattern)

            # Normalize coordinates by grid size
            grid_w, grid_h = self.som.grid_size
            normalized_x = winner_neuron[0] / (grid_w - 1 if grid_w > 1 else 1)
            normalized_y = winner_neuron[1] / (grid_h - 1 if grid_h > 1 else 1)

            batch_features.append([normalized_x, normalized_y])

        return torch.tensor(batch_features, dtype=torch.float32).to(device)
