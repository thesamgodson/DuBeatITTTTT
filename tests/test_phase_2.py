import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market_mind.participants import InformedTraderModule, NoiseTraderModule, MarketMakerModule

class TestParticipantModules(unittest.TestCase):

    def setUp(self):
        """Set up common variables for tests."""
        self.batch_size = 4
        self.seq_length = 20
        self.feature_dim = 3  # Assuming each module looks at 3 relevant features
        self.hidden_dim = 16

        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.feature_dim)

    def _test_module(self, module_class, module_name):
        """A generic test function for any participant module."""
        print(f"\n--- Testing {module_name} ---")

        # 1. Test initialization
        module = module_class(feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)

        # 2. Test forward pass
        output = module(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, 1), f"{module_name} output shape is incorrect.")
        self.assertFalse(torch.isnan(output).any(), f"{module_name} output contains NaN values.")
        print(f"✓ {module_name} forward pass successful.")

        # 3. Test trainability
        loss = output.sum()
        loss.backward()

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter '{name}' in {module_name} has no gradient.")
        print(f"✓ {module_name} is trainable.")

    def test_informed_trader_module(self):
        """Tests the InformedTraderModule."""
        self._test_module(InformedTraderModule, "InformedTraderModule")

    def test_noise_trader_module(self):
        """Tests the NoiseTraderModule."""
        self._test_module(NoiseTraderModule, "NoiseTraderModule")

    def test_market_maker_module(self):
        """Tests the MarketMakerModule."""
        self._test_module(MarketMakerModule, "MarketMakerModule")

if __name__ == '__main__':
    unittest.main()
