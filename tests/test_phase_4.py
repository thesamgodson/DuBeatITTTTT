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

from src.data_processing.pipeline import MarketDataReader, Aligner
from src.market_mind.model import MarketMindModel

class TestPhase4Performance(unittest.TestCase):

    def test_market_mind_performance(self):
        """
        Full end-to-end performance test for the MarketMindModel.
        """
        print("\n\n--- Running Phase 4: MarketMind Performance Test ---")

        # 1. --- Data Loading and Full Feature Creation ---
        print("\n[1/4] Loading and preparing data...")
        reader = MarketDataReader(assets=['BTC-USD'], start_date='2018-01-01', end_date='2023-12-31')
        processed_data = reader.load_and_process()
        btc_data = processed_data['BTC-USD']

        backcast_length = 30
        forecast_horizon = 1
        aligner = Aligner(backcast_length=backcast_length, forecast_horizon=forecast_horizon)

        X_full, y_full = aligner.create_sequences(btc_data, target_column='LogReturns')
        print("✓ Full dataset created.")

        # 2. --- Splitting Features for Each Model Input Branch ---
        print("\n[2/4] Splitting features for model inputs...")
        all_feature_names = [col for col in btc_data.columns if col not in ['Date', 'LogReturns']]

        feature_map = {
            'regime': ['Volatility', 'ATR'],
            'informed': ['InformedTradingProxy', 'Returns'],
            'noise': ['NoiseTradingProxy'],
            'mm': ['LiquidityProxy', 'High', 'Low', 'Open', 'Close', 'Volume']
        }

        feature_indices = {key: [all_feature_names.index(f) for f in features] for key, features in feature_map.items()}

        X_regime = X_full[:, :, feature_indices['regime']]
        X_informed = X_full[:, :, feature_indices['informed']]
        X_noise = X_full[:, :, feature_indices['noise']]
        X_mm = X_full[:, :, feature_indices['mm']]
        print("✓ Features successfully split.")

        # 3. --- Data Scaling and Batching ---
        print("\n[3/4] Scaling data and creating DataLoaders...")
        train_size = int(len(X_full) * 0.8)

        X_regime_train, X_regime_test = X_regime[:train_size], X_regime[train_size:]
        X_informed_train, X_informed_test = X_informed[:train_size], X_informed[train_size:]
        X_noise_train, X_noise_test = X_noise[:train_size], X_noise[train_size:]
        X_mm_train, X_mm_test = X_mm[:train_size], X_mm[train_size:]
        y_train, y_test = y_full[:train_size], y_full[train_size:]

        scalers = {
            'regime': StandardScaler().fit(X_regime_train.reshape(-1, X_regime_train.shape[-1])),
            'informed': StandardScaler().fit(X_informed_train.reshape(-1, X_informed_train.shape[-1])),
            'noise': StandardScaler().fit(X_noise_train.reshape(-1, X_noise_train.shape[-1])),
            'mm': StandardScaler().fit(X_mm_train.reshape(-1, X_mm_train.shape[-1])),
            'y': StandardScaler().fit(y_train)
        }

        X_regime_train_s = scalers['regime'].transform(X_regime_train.reshape(-1, X_regime_train.shape[-1])).reshape(X_regime_train.shape)
        X_informed_train_s = scalers['informed'].transform(X_informed_train.reshape(-1, X_informed_train.shape[-1])).reshape(X_informed_train.shape)
        X_noise_train_s = scalers['noise'].transform(X_noise_train.reshape(-1, X_noise_train.shape[-1])).reshape(X_noise_train.shape)
        X_mm_train_s = scalers['mm'].transform(X_mm_train.reshape(-1, X_mm_train.shape[-1])).reshape(X_mm_train.shape)
        y_train_s = scalers['y'].transform(y_train)

        X_regime_test_s = scalers['regime'].transform(X_regime_test.reshape(-1, X_regime_test.shape[-1])).reshape(X_regime_test.shape)
        X_informed_test_s = scalers['informed'].transform(X_informed_test.reshape(-1, X_informed_test.shape[-1])).reshape(X_informed_test.shape)
        X_noise_test_s = scalers['noise'].transform(X_noise_test.reshape(-1, X_noise_test.shape[-1])).reshape(X_noise_test.shape)
        X_mm_test_s = scalers['mm'].transform(X_mm_test.reshape(-1, X_mm_test.shape[-1])).reshape(X_mm_test.shape)

        train_dataset = TensorDataset(
            torch.tensor(X_informed_train_s, dtype=torch.float32),
            torch.tensor(X_noise_train_s, dtype=torch.float32),
            torch.tensor(X_mm_train_s, dtype=torch.float32),
            torch.tensor(X_regime_train_s, dtype=torch.float32),
            torch.tensor(y_train_s, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        print("✓ Data scaling and batching complete.")

        # 4. --- Model Training & Evaluation ---
        print("\n[4/4] Training and evaluating MarketMind model...")
        model = MarketMindModel(
            informed_trader_features=len(feature_map['informed']),
            noise_trader_features=len(feature_map['noise']),
            market_maker_features=len(feature_map['mm']),
            regime_features=len(feature_map['regime']),
            hidden_dim=32
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        epochs = 100
        for epoch in range(epochs):
            model.train()
            for x_inf, x_noise, x_mm, x_reg, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred_scaled = model(x_inf, x_noise, x_mm, x_reg)
                loss = loss_fn(y_pred_scaled, y_batch)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            x_inf_test_t = torch.tensor(X_informed_test_s, dtype=torch.float32)
            x_noise_test_t = torch.tensor(X_noise_test_s, dtype=torch.float32)
            x_mm_test_t = torch.tensor(X_mm_test_s, dtype=torch.float32)
            x_reg_test_t = torch.tensor(X_regime_test_s, dtype=torch.float32)

            y_pred_scaled = model(x_inf_test_t, x_noise_test_t, x_mm_test_t, x_reg_test_t)
            y_pred_returns = scalers['y'].inverse_transform(y_pred_scaled.numpy())

        correct_direction = (np.sign(y_pred_returns) == np.sign(y_test)).sum()
        directional_accuracy = (correct_direction / len(y_test)) * 100

        print(f"\n--- Final Performance ---")
        print(f"  ✓ Final Directional Accuracy: {directional_accuracy:.2f}%")

        self.assertGreater(directional_accuracy, 65.0, "Directional accuracy should be > 65%.")

if __name__ == '__main__':
    unittest.main()
