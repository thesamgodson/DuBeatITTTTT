import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .dynamic_som import DynamicSOM
from .novelty_detector import ForecastDivergenceDetector
from sklearn.cluster import DBSCAN

class DriftNet(nn.Module):
    """
    DriftNet v0.2: A model that uses a Forecast Divergence Detector (FDD)
    to trigger the growth of a dynamic SOM and its expert ensemble.
    """
    def __init__(self, input_dim: int, forecast_horizon: int, initial_experts: int = 4):
        """
        Initializes the DriftNet model.
        """
        super().__init__()
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon

        self.dynamic_som = DynamicSOM(input_dim=input_dim, initial_nodes=initial_experts)
        self.fdd = ForecastDivergenceDetector()

        self.experts = nn.ModuleList([
            nn.Linear(input_dim, forecast_horizon) for _ in range(initial_experts)
        ])

        self.birth_log = []

    def _spawn_new_expert(self, bmu_idx: int, timestamp: pd.Timestamp, trigger_input: np.array):
        """Adds a new expert to the ensemble and logs the birth event."""
        new_node_idx = self.dynamic_som.add_node(bmu_idx)
        device = next(self.parameters()).device
        new_expert = nn.Linear(self.input_dim, self.forecast_horizon).to(device)
        self.experts.append(new_expert)

        self.birth_log.append({
            "timestamp": timestamp, "new_expert_id": new_node_idx,
            "spawned_from_expert_id": bmu_idx, "trigger_input_sample": trigger_input
        })
        assert len(self.experts) == self.dynamic_som.num_nodes, "Expert count mismatch!"

    def check_and_adapt(self, forecast_error: float, x: torch.Tensor, timestamp: pd.Timestamp):
        """
        Checks for novelty using the FDD and adapts the model if necessary.
        This is called from the main evaluation loop.
        """
        if self.fdd.check_novelty(forecast_error):
            x_numpy = x.detach().cpu().numpy().flatten()
            bmu_to_spawn_from = self.dynamic_som.find_bmu(x_numpy)
            self._spawn_new_expert(bmu_to_spawn_from, timestamp, x_numpy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass now only handles forecasting, not adaptation.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x_numpy = x.detach().cpu().numpy().flatten()
        bmu_idx = self.dynamic_som.find_bmu(x_numpy)
        active_expert = self.experts[bmu_idx]
        forecast = active_expert(x)

        return forecast

    def get_bmu_and_update_som(self, x: torch.Tensor, epoch: int, max_epochs: int) -> int:
        """
        Finds the BMU and updates the SOM weights.
        This should be called during the training loop.
        """
        x_numpy = x.detach().cpu().numpy().flatten()
        bmu_idx = self.dynamic_som.find_bmu(x_numpy)
        self.dynamic_som.update(x_numpy, bmu_idx, epoch, max_epochs)
        return bmu_idx

    def _merge_experts(self, eps: float = 0.5, min_samples: int = 2):
        """
        Finds and merges experts that have become redundant (i.e., their weight
        vectors are very similar).

        Args:
            eps (float): The maximum distance between two samples for one to be
                         considered as in the neighborhood of the other (for DBSCAN).
            min_samples (int): The number of samples in a neighborhood for a point
                               to be considered as a core point.
        """
        if len(self.experts) <= min_samples:
            return # Not enough experts to merge

        print(f"Running expert merging check on {len(self.experts)} experts...")

        # Get all expert weight vectors
        expert_weights = np.array([exp.weight.detach().cpu().numpy().flatten() for exp in self.experts])

        # Use DBSCAN to find clusters of similar experts
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(expert_weights)
        labels = clustering.labels_

        # Find clusters (label > -1)
        unique_labels = set(labels)
        unique_labels.discard(-1) # -1 is for noise points, not a cluster

        if not unique_labels:
            print("No clusters found to merge.")
            return

        experts_to_remove = []
        for k in unique_labels:
            cluster_indices = list(np.where(labels == k)[0])
            print(f"Found cluster of {len(cluster_indices)} experts to merge: {cluster_indices}")

            # 1. Create a new merged expert (average the weights and biases)
            avg_weight = torch.mean(torch.stack([self.experts[i].weight for i in cluster_indices]), dim=0)
            avg_bias = torch.mean(torch.stack([self.experts[i].bias for i in cluster_indices]), dim=0)

            merged_expert = nn.Linear(self.input_dim, self.forecast_horizon)
            merged_expert.weight = nn.Parameter(avg_weight)
            merged_expert.bias = nn.Parameter(avg_bias)

            # 2. Add the new merged expert to the ensemble
            self.experts.append(merged_expert.to(next(self.parameters()).device))

            # 3. Average the SOM node weights and positions for the merged cluster
            avg_som_weight = np.mean([self.dynamic_som.codebook[i] for i in cluster_indices], axis=0)
            avg_som_position = np.mean([self.dynamic_som.positions[i] for i in cluster_indices], axis=0)
            self.dynamic_som.codebook.append(avg_som_weight)
            self.dynamic_som.positions.append(avg_som_position)
            self.dynamic_som.num_nodes += 1

            # 4. Mark the old experts for removal
            experts_to_remove.extend(cluster_indices)

        # 5. Remove the old experts and SOM nodes
        # We must iterate backwards to not mess up the indices
        for i in sorted(list(set(experts_to_remove)), reverse=True):
            del self.experts[i]
            del self.dynamic_som.codebook[i]
            del self.dynamic_som.positions[i]
            self.dynamic_som.num_nodes -= 1

        print(f"Merging complete. New expert count: {len(self.experts)}")
