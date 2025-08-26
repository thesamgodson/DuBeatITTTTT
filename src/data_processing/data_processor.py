import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd

class TimeSeriesDataLoader:
    def __init__(self, df: pd.DataFrame, sequence_length: int, forecast_horizon: int, train_split: float = 0.8, batch_size: int = 32, feature_cols=['Close']):
        self.df = df
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.train_split = train_split
        self.batch_size = batch_size
        self.feature_cols = feature_cols

        self._process_data()

    def _process_data(self):
        # Keep only the feature columns
        data = self.df[self.feature_cols].values.astype(np.float32)

        # Create sequences
        X, y = self._create_sequences(data)

        # Squeeze the last dimension if only one feature is used
        if len(self.feature_cols) == 1:
            X = X.squeeze(-1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def _create_sequences(self, data):
        X, y = [], []
        num_sequences = len(data) - self.sequence_length - self.forecast_horizon + 1
        for i in range(num_sequences):
            seq_end = i + self.sequence_length
            forecast_end = seq_end + self.forecast_horizon

            # Use all feature columns for the input sequence
            X.append(data[i:seq_end])

            # Use the first feature column (assumed to be 'Close') for the target
            y.append(data[seq_end:forecast_end, 0])

        return np.array(X), np.array(y)

    def get_loaders(self):
        dataset = TensorDataset(self.X, self.y)

        # Split data
        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size

        if train_size == 0 or test_size == 0:
            raise ValueError("Dataset is too small to be split. Please provide more data or adjust the train/test split ratio.")

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def get_full_dataset(self):
        return TensorDataset(self.X, self.y)
