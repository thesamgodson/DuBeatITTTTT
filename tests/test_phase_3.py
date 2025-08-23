import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nbeats.blocks import TrendBlock, SeasonalityBlock, GenericBlock
from src.nbeats.model import PureNBEATS
from src.data_processing.data_loader import load_bitcoin_data

class TestPhase3NBEATSImplementation(unittest.TestCase):

    def test_nbeats_blocks(self):
        """
        Validation test for the individual N-BEATS blocks.
        """
        print("\n\n--- Running N-BEATS Block Shape Test ---")
        batch_size = 32
        backcast_length = 30
        forecast_length = 1
        units = 64
        thetas_dim = 3

        block = TrendBlock(units=units, thetas_dim=thetas_dim,
                           backcast_length=backcast_length, forecast_length=forecast_length)

        x = torch.randn(batch_size, backcast_length)
        backcast, forecast = block(x)

        self.assertEqual(backcast.shape, (batch_size, backcast_length))
        self.assertEqual(forecast.shape, (batch_size, forecast_length))
        self.assertFalse(torch.isnan(forecast).any())
        print("✓ N-BEATS blocks working correctly")

    def test_pure_nbeats_performance(self):
        """
        Performance test for the PureNBEATS model.
        Trains the model and checks if RMSE is below the target.
        """
        print("\n--- Running Pure N-BEATS Performance Test ---")
        # 1. --- Data Preparation ---
        backcast_length = 30
        forecast_length = 1

        # Load data
        df = load_bitcoin_data(start_date='2018-01-01', end_date='2023-12-31')
        close_prices = df['Close'].values

        # Normalize data
        scaler_min = close_prices.min()
        scaler_max = close_prices.max()
        scaled_prices = (close_prices - scaler_min) / (scaler_max - scaler_min)

        # Create sliding windows
        X, y = [], []
        for i in range(len(scaled_prices) - backcast_length - forecast_length + 1):
            X.append(scaled_prices[i : i + backcast_length])
            y.append(scaled_prices[i + backcast_length : i + backcast_length + forecast_length])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 2. --- Model Training ---
        model = PureNBEATS(backcast_length=backcast_length, forecast_length=forecast_length, units=512)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        epochs = 40  # A few epochs to ensure learning happens
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # 3. --- Evaluation ---
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test)

        # Inverse transform to original scale
        y_pred = y_pred_scaled.numpy() * (scaler_max - scaler_min) + scaler_min
        y_true = y_test.numpy() * (scaler_max - scaler_min) + scaler_min

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        print(f"✓ Pure N-BEATS Test RMSE: {rmse:.1f}")

        # 4. --- Assertion ---
        self.assertLess(rmse, 800, "Pure N-BEATS RMSE should be less than 800.")

if __name__ == '__main__':
    unittest.main()
