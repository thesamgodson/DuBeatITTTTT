import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.som_analysis.feature_extractor import SOMFeatureExtractor
from src.nbeats.model import NBEATSWithSOM

class TestBaselineReproduction(unittest.TestCase):

    def test_reproduce_baseline_performance(self):
        """
        Tests if we can reproduce the ~54% directional accuracy of the NBEATSWithSOM model.
        """
        print("\n\n--- Reproducing NBEATSWithSOM Baseline Performance ---")

        # 1. --- Configuration and Data Loading ---
        backcast_length = 30
        forecast_horizon = 1

        df = load_bitcoin_data(start_date='2018-01-01', end_date='2023-12-31')
        prices = df['Close'].values

        train_size = int(len(prices) * 0.8)
        train_prices = prices[:train_size]

        # 2. --- Train SOMs and Pre-compute Features ---
        som_configs = {
            '3d': {'window_size': 3, 'grid_size': (5, 5)},
            '7d': {'window_size': 7, 'grid_size': (7, 7)},
            '21d': {'window_size': 21, 'grid_size': (10, 10)},
        }
        # Train on the training data only
        feature_extractor = SOMFeatureExtractor(som_configs).train(train_prices)

        all_som_features = []
        max_som_window = max(c['window_size'] for c in som_configs.values())
        for i in range(len(prices) - max_som_window + 1):
            sequence = prices[:i + max_som_window]
            all_som_features.append(feature_extractor.extract_features(sequence))
        all_som_features = np.array(all_som_features)

        # 3. --- Create Full Aligned Dataset ---
        X_ts, X_som, y = [], [], []
        start_t = max(backcast_length - 1, max_som_window - 1)
        end_t = len(prices) - forecast_horizon

        for t in range(start_t, end_t):
            ts_window = prices[t - (backcast_length - 1) : t + 1]
            y_return = (prices[t + forecast_horizon] - prices[t]) / prices[t]

            som_feature_index = t - (max_som_window - 1)
            som_feature_window = all_som_features[som_feature_index]

            X_ts.append(ts_window)
            X_som.append(som_feature_window)
            y.append(y_return)

        X_ts = np.array(X_ts)
        X_som = np.array(X_som)
        y = np.array(y).reshape(-1, 1)

        # 4. --- Data Scaling and Batching ---
        train_size_final = int(len(X_ts) * 0.8)

        X_ts_train, X_ts_test = X_ts[:train_size_final], X_ts[train_size_final:]
        X_som_train, X_som_test = X_som[:train_size_final], X_som[train_size_final:]
        y_train, y_test = y[:train_size_final], y[train_size_final:]

        ts_scaler = StandardScaler().fit(X_ts_train)
        X_ts_train_s = ts_scaler.transform(X_ts_train)
        X_ts_test_s = ts_scaler.transform(X_ts_test)

        return_scaler = StandardScaler().fit(y_train)
        y_train_s = return_scaler.transform(y_train)

        train_dataset = TensorDataset(
            torch.tensor(X_ts_train_s, dtype=torch.float32),
            torch.tensor(X_som_train, dtype=torch.float32), # SOM features are already 0-1
            torch.tensor(y_train_s, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # 5. --- Model Training ---
        model = NBEATSWithSOM(
            backcast_length=backcast_length,
            forecast_length=forecast_horizon,
            som_feature_dim=X_som.shape[1],
            units=512
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        epochs = 80
        for epoch in range(epochs):
            for x_ts_batch, x_som_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred_scaled, _ = model(x_ts_batch, x_som_batch)
                loss = loss_fn(y_pred_scaled, y_batch)
                loss.backward()
                optimizer.step()

        # 6. --- Evaluation ---
        model.eval()
        with torch.no_grad():
            x_ts_test_t = torch.tensor(X_ts_test_s, dtype=torch.float32)
            x_som_test_t = torch.tensor(X_som_test, dtype=torch.float32)

            y_pred_scaled, _ = model(x_ts_test_t, x_som_test_t)
            y_pred_returns = return_scaler.inverse_transform(y_pred_scaled.numpy())

        correct_direction = (np.sign(y_pred_returns) == np.sign(y_test)).sum()
        directional_accuracy = (correct_direction / len(y_test)) * 100

        print(f"\n--- Baseline Reproduced Performance ---")
        print(f"  ✓ Final Directional Accuracy: {directional_accuracy:.2f}%")

        self.assertGreater(directional_accuracy, 53.0, "Reproduced accuracy should be > 53%.")
        self.assertLess(directional_accuracy, 55.0, "Reproduced accuracy should be < 55%.")

if __name__ == '__main__':
    unittest.main()
