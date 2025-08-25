import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.som_analysis.feature_extractor import SOMFeatureExtractor
from src.nbeats.model import NBEATSWithFeatures
from src.feature_engineering.trends import fetch_google_trends

class TestPerformanceWithTrends(unittest.TestCase):

    def test_performance_with_google_trends(self):
        """
        Tests the full model with the addition of Google Trends data.
        """
        print("\n\n--- Testing Performance with Google Trends Integration ---")

        # 1. --- Data Loading and Merging ---
        print("\n[1/5] Loading and fetching all data sources...")
        start_date = '2018-01-01'
        end_date = '2023-12-31'

        price_df = load_bitcoin_data(start_date=start_date, end_date=end_date)
        trends_df = fetch_google_trends('Bitcoin', start_date, end_date)

        # Merge the datasets
        self.assertFalse(price_df.empty, "Price data failed to load.")
        self.assertFalse(trends_df.empty, "Trends data failed to load.")
        merged_df = pd.merge(price_df, trends_df, on='Date', how='inner')
        print("✓ Price and Trends data successfully loaded and merged.")

        # 2. --- Feature and Dataset Creation ---
        print("\n[2/5] Creating features and aligned dataset...")
        backcast_length = 30
        forecast_horizon = 1
        prices = merged_df['Close'].values

        # Create SOM features
        som_configs = {'3d': {'window_size': 3, 'grid_size': (5, 5)}}
        feature_extractor = SOMFeatureExtractor(som_configs).train(prices)

        all_som_features, all_ts_features, all_trends_features, y = [], [], [], []

        max_som_window = max(c['window_size'] for c in som_configs.values())
        start_t = max(backcast_length - 1, max_som_window - 1)
        end_t = len(merged_df) - forecast_horizon

        for t in range(start_t, end_t):
            # Time series and SOM features
            ts_window = prices[t - (backcast_length - 1) : t + 1]
            all_ts_features.append(ts_window)

            som_seq = prices[:t + 1]
            all_som_features.append(feature_extractor.extract_features(som_seq))

            # Trends feature
            trends_window = merged_df['trends_score'].values[t - (backcast_length - 1) : t + 1]
            all_trends_features.append(trends_window)

            # Target
            y_return = (prices[t + forecast_horizon] - prices[t]) / prices[t]
            y.append(y_return)

        X_ts = np.array(all_ts_features)
        X_som = np.array(all_som_features)
        X_trends = np.array(all_trends_features).reshape(-1, backcast_length, 1)
        y = np.array(y).reshape(-1, 1)
        print("✓ All feature streams created.")

        # 3. --- Data Scaling and Batching ---
        print("\n[3/5] Scaling and batching data...")
        train_size = int(len(X_ts) * 0.8)

        # Split data
        X_ts_train, X_ts_test = X_ts[:train_size], X_ts[train_size:]
        X_som_train, X_som_test = X_som[:train_size], X_som[train_size:]
        X_trends_train, X_trends_test = X_trends[:train_size], X_trends[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Scale data
        ts_scaler = StandardScaler().fit(X_ts_train)
        trends_scaler = StandardScaler().fit(X_trends_train.reshape(-1, 1))
        return_scaler = StandardScaler().fit(y_train)

        X_ts_train_s = ts_scaler.transform(X_ts_train)
        X_trends_train_s = trends_scaler.transform(X_trends_train.reshape(-1, 1)).reshape(X_trends_train.shape)
        y_train_s = return_scaler.transform(y_train)

        # Create dataset and loader
        train_dataset = TensorDataset(
            torch.tensor(X_ts_train_s, dtype=torch.float32),
            torch.tensor(X_som_train, dtype=torch.float32),
            torch.tensor(X_trends_train_s, dtype=torch.float32),
            torch.tensor(y_train_s, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        print("✓ Data pipeline complete.")

        # 4. --- Model Training ---
        print("\n[4/5] Training model with integrated features...")
        model = NBEATSWithFeatures(
            backcast_length=backcast_length,
            forecast_length=forecast_horizon,
            som_feature_dim=X_som.shape[1],
            external_feature_dim=X_trends.shape[2],
            units=256
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        epochs = 80
        for epoch in range(epochs):
            for x_ts_b, x_som_b, x_ext_b, y_b in train_loader:
                optimizer.zero_grad()
                y_pred_s, _ = model(x_ts_b, x_som_b, x_ext_b)
                loss = loss_fn(y_pred_s, y_b)
                loss.backward()
                optimizer.step()
        print("✓ Model training complete.")

        # 5. --- Evaluation ---
        print("\n[5/5] Evaluating final performance...")
        model.eval()
        with torch.no_grad():
            X_ts_test_s = ts_scaler.transform(X_ts_test)
            X_trends_test_s = trends_scaler.transform(X_trends_test.reshape(-1, 1)).reshape(X_trends_test.shape)

            y_pred_s, _ = model(
                torch.tensor(X_ts_test_s, dtype=torch.float32),
                torch.tensor(X_som_test, dtype=torch.float32),
                torch.tensor(X_trends_test_s, dtype=torch.float32)
            )
            y_pred = return_scaler.inverse_transform(y_pred_s.numpy())

        correct = (np.sign(y_pred) == np.sign(y_test)).sum()
        accuracy = (correct / len(y_test)) * 100

        print(f"\n--- Final Performance with Trends ---")
        print(f"  ✓ Directional Accuracy: {accuracy:.2f}%")

        self.assertGreater(accuracy, 55.0, "Accuracy with trends should be > 55%.")

if __name__ == '__main__':
    unittest.main()
