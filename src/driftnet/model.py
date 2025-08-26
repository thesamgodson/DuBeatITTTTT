import torch
import torch.nn as nn
import numpy as np

from .dynamic_som import DynamicSOM
from .novelty_detector import RegimeNoveltyDetector

class DriftNet(nn.Module):
    """
    DriftNet v0.1: A model that combines a dynamic SOM for regime detection
    with an ensemble of simple expert models for forecasting.
    """
    def __init__(self, input_dim: int, forecast_horizon: int, initial_experts: int = 4):
        """
        Initializes the DriftNet model.

        Args:
            input_dim (int): The dimensionality of the input sequences.
            forecast_horizon (int): The number of time steps to forecast.
            initial_experts (int): The number of initial experts/SOM nodes.
        """
        super().__init__()
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon

        # 1. Core Components
        self.dynamic_som = DynamicSOM(input_dim=input_dim, initial_nodes=initial_experts)
        self.novelty_detector = RegimeNoveltyDetector(input_dim=input_dim, latent_dim=10)

        # 2. Expert Ensemble
        # Each expert is a simple linear model for this version.
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, forecast_horizon) for _ in range(initial_experts)
        ])

    def _spawn_new_expert(self, bmu_idx: int):
        """
        Adds a new expert to the ensemble, linked to a new SOM node.

        Args:
            bmu_idx (int): The index of the BMU that triggered the novelty event.
        """
        # Add a new node to the SOM near the BMU
        new_node_idx = self.dynamic_som.add_node(bmu_idx)

        # Create a new expert and add it to the list
        # Ensure the new expert is on the same device as the rest of the model
        device = next(self.parameters()).device
        new_expert = nn.Linear(self.input_dim, self.forecast_horizon).to(device)
        self.experts.append(new_expert)

        # Ensure the number of experts matches the number of SOM nodes
        assert len(self.experts) == self.dynamic_som.num_nodes, "Expert count mismatch!"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for a single time step's input sequence.

        Args:
            x (torch.Tensor): A single input sequence, shape (1, input_dim).

        Returns:
            torch.Tensor: The forecast from the selected expert.
        """
        # Ensure input is flattened
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x_numpy = x.detach().cpu().numpy().flatten()

        # 1. Check for novelty
        if self.novelty_detector.check_novelty(x):
            # Find the BMU to spawn a new expert near
            bmu_to_spawn_from = self.dynamic_som.find_bmu(x_numpy)
            self._spawn_new_expert(bmu_to_spawn_from)

        # 2. Find the current regime (BMU)
        bmu_idx = self.dynamic_som.find_bmu(x_numpy)

        # 3. Select the corresponding expert
        active_expert = self.experts[bmu_idx]

        # 4. Generate forecast
        forecast = active_expert(x)

        return forecast

    def train_novelty_detector(self, data_loader, epochs=5):
        """
        Convenience method to train the internal novelty detector.
        """
        self.novelty_detector.train_autoencoder(data_loader, epochs=epochs)

    def get_bmu_and_update_som(self, x: torch.Tensor, epoch: int, max_epochs: int) -> int:
        """
        Finds the BMU and updates the SOM weights.
        This should be called during the training loop.
        """
        x_numpy = x.detach().cpu().numpy().flatten()
        bmu_idx = self.dynamic_som.find_bmu(x_numpy)
        self.dynamic_som.update(x_numpy, bmu_idx, epoch, max_epochs)
        return bmu_idx
