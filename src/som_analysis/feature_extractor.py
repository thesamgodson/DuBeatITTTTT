import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.som_analysis.som_trainer import SimpleSOM
from src.data_processing.feature_engineer import create_pattern_vectors

class SOMFeatureExtractor:
    """
    Manages the training of multiple SOMs and extracts coordinate-based features.
    """
    def __init__(self, som_configs: dict):
        self.som_configs = som_configs
        self.soms = {}

    def train(self, price_data: np.ndarray):
        """
        Trains all configured SOMs on the historical price data.
        """
        for name, config in self.som_configs.items():
            window_size = config['window_size']
            grid_size = config['grid_size']

            patterns = create_pattern_vectors(price_data, window_size=window_size)

            if patterns.shape[0] == 0:
                continue

            som = SimpleSOM(grid_size=grid_size, input_len=window_size)
            som.train(patterns, num_iteration=500)

            self.soms[name] = som

        return self

    def extract_features(self, price_sequence: np.ndarray) -> np.ndarray:
        """
        Extracts SOM winner coordinates for a given price sequence.
        """
        features = []
        for name, som in self.soms.items():
            window_size = self.som_configs[name]['window_size']
            grid_size = self.som_configs[name]['grid_size']

            if len(price_sequence) < window_size:
                raise ValueError(f"Price sequence length {len(price_sequence)} is too short for window size {window_size}")

            pattern = price_sequence[-window_size:]
            normalized_pattern = pattern / (pattern[0] + 1e-9)
            winner_neuron = som.winner(normalized_pattern)

            # Normalize coordinates by grid size
            normalized_x = winner_neuron[0] / (grid_size[0] - 1)
            normalized_y = winner_neuron[1] / (grid_size[1] - 1)

            features.extend([normalized_x, normalized_y])

        return np.array(features, dtype=np.float32)
