import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market_mind.participants import RegimeDetectionModule
from src.market_mind.model import MarketMindModel

class TestPhase3ModelIntegration(unittest.TestCase):

    def setUp(self):
        """Set up common variables for tests."""
        self.batch_size = 4
        self.seq_length = 20
        self.hidden_dim = 32

    def test_regime_detection_module(self):
        """
        Tests the RegimeDetectionModule in isolation.
        """
        print("\n\n--- Testing Regime Detection Module ---")
        feature_dim = 2 # e.g., Volatility and ATR
        module = RegimeDetectionModule(feature_dim=feature_dim, hidden_dim=self.hidden_dim)
        sample_input = torch.randn(self.batch_size, self.seq_length, feature_dim)

        # Test forward pass
        output = module(sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        print("✓ Forward pass successful and output shape is correct.")

        # Test trainability
        loss = output.sum()
        loss.backward()
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter '{name}' has no gradient.")
        print("✓ Module is trainable.")

    def test_market_mind_model_integration(self):
        """
        Tests the full MarketMindModel to ensure all components are integrated correctly.
        """
        print("\n--- Testing Full MarketMind Model Integration ---")
        # Define feature dimensions for each input branch
        feature_dims = {
            'informed': 3,
            'noise': 2,
            'mm': 2,
            'regime': 2
        }

        # Instantiate the full model
        model = MarketMindModel(
            informed_trader_features=feature_dims['informed'],
            noise_trader_features=feature_dims['noise'],
            market_maker_features=feature_dims['mm'],
            regime_features=feature_dims['regime'],
            hidden_dim=self.hidden_dim
        )

        # Create sample data for each input branch
        x_informed = torch.randn(self.batch_size, self.seq_length, feature_dims['informed'])
        x_noise = torch.randn(self.batch_size, self.seq_length, feature_dims['noise'])
        x_mm = torch.randn(self.batch_size, self.seq_length, feature_dims['mm'])
        x_regime = torch.randn(self.batch_size, self.seq_length, feature_dims['regime'])

        # Perform a forward and backward pass
        prediction = model(x_informed, x_noise, x_mm, x_regime)
        loss = prediction.sum()
        loss.backward()

        # 1. Check final output shape
        self.assertEqual(prediction.shape, (self.batch_size, 1), "Final prediction shape is incorrect.")
        print("✓ Forward pass successful and output shape is correct.")

        # 2. Check trainability of all sub-modules
        # We check one representative parameter from each sub-module
        self.assertIsNotNone(model.informed_trader_module.output_layer.weight.grad)
        self.assertIsNotNone(model.noise_trader_module.output_layer.weight.grad)
        self.assertIsNotNone(model.market_maker_module.output_layer.weight.grad)
        self.assertIsNotNone(model.regime_detection_module.output_layer.weight.grad)
        self.assertIsNotNone(model.attention.in_proj_weight.grad)
        self.assertIsNotNone(model.output_layer.weight.grad)
        print("✓ All sub-modules are connected and trainable.")

if __name__ == '__main__':
    unittest.main()
