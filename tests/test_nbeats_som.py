import unittest
import numpy as np
import pandas as pd
import torch
from src.data_processing.data_processor import TimeSeriesDataLoader
from src.som_analysis.som_trainer import SimpleSOM
from src.som_analysis.feature_extractor import SOMFeatureExtractor
from src.nbeats.nbeats_model import NBEATS, NBEATSWithSOM

class TestNBEATSSOMIntegration(unittest.TestCase):

    def setUp(self):
        # Create a dummy data file
        self.data_path = 'dummy_data.csv'
        with open(self.data_path, 'w') as f:
            f.write("timestamp,Close,Volume\n")
            for i in range(500):
                f.write(f"{i},{100 + 5 * np.sin(i / 10)},{1000 + 100 * np.random.rand()}\n")

        # Load data into a DataFrame
        df = pd.read_csv(self.data_path)

        # 1. Data Loader
        self.data_loader = TimeSeriesDataLoader(
            df=df,
            sequence_length=30,
            forecast_horizon=5,
            train_split=0.8,
            batch_size=16,
            feature_cols=['Close']
        )
        self.train_loader, self.test_loader = self.data_loader.get_loaders()

        # 2. SOM
        sample_data = next(iter(self.train_loader))[0].numpy()
        self.som = SimpleSOM(input_len=30, grid_size=(10, 10), learning_rate=0.5, sigma=1.0)
        self.som.train(sample_data, num_iteration=100)

        # 3. Feature Extractor
        self.feature_extractor = SOMFeatureExtractor(self.som)

        # 4. NBEATS with SOM model
        self.nbeats_som_model = NBEATSWithSOM(
            som_feature_extractor=self.feature_extractor,
            sequence_length=30,
            forecast_horizon=5,
            n_stacks=2,
            n_blocks=3,
            n_layers=4,
            layer_width=128
        )

    def test_pipeline_forward_pass(self):
        """Test a full forward pass through the NBEATSWithSOM model."""
        x, y_true = next(iter(self.train_loader))

        self.assertEqual(x.shape[0], 16) # Batch size
        self.assertEqual(x.shape[1], 30) # Sequence length
        self.assertEqual(y_true.shape[1], 5) # Forecast horizon

        # Forward pass
        y_pred = self.nbeats_som_model(x)

        # Check output shape
        self.assertEqual(y_pred.shape, y_true.shape)
        self.assertFalse(torch.isnan(y_pred).any(), "Output contains NaNs")

    def test_training_loop(self):
        """Test a minimal training loop."""
        optimizer = torch.optim.Adam(self.nbeats_som_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        x, y_true = next(iter(self.train_loader))

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = self.nbeats_som_model(x)

        # Calculate loss
        loss = criterion(y_pred, y_true)
        self.assertTrue(loss.item() > 0)

        # Backward pass and optimization
        loss.backward()

        # Check for gradients
        has_grads = any(p.grad is not None for p in self.nbeats_som_model.parameters() if p.requires_grad)
        self.assertTrue(has_grads, "No gradients were computed")

        optimizer.step()

    def tearDown(self):
        import os
        os.remove(self.data_path)

if __name__ == '__main__':
    unittest.main()
