import unittest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.som_analysis.som_trainer import SimpleSOM
from src.som_analysis.pattern_analyzer import analyze_som_patterns

class TestPhase2SOMImplementation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up data and train a SOM once for all tests."""
        # Create distinct patterns: 5 upward, 5 downward
        upward_patterns = np.array([
            [1.0, 1.1, 1.2], [1.0, 1.2, 1.3], [1.0, 1.05, 1.15],
            [1.0, 1.1, 1.25], [1.0, 1.15, 1.3]
        ])
        downward_patterns = np.array([
            [1.0, 0.9, 0.8], [1.0, 0.8, 0.7], [1.0, 0.95, 0.85],
            [1.0, 0.9, 0.75], [1.0, 0.85, 0.7]
        ])
        cls.patterns = np.vstack([upward_patterns, downward_patterns])

        # Assign positive returns to upward patterns, negative to downward
        cls.future_returns = np.array([0.02, 0.03, 0.025, 0.028, 0.032,
                                     -0.02, -0.03, -0.025, -0.028, -0.032])

        # Initialize and train the SOM
        cls.som = SimpleSOM(grid_size=(10, 10), input_len=3)
        cls.som.train(cls.patterns, num_iteration=100)

    def test_som_training(self):
        """
        Validation test for SOM training.
        """
        # 1. Check if quantization error is reasonable
        error = self.som.get_quantization_error(self.patterns)
        self.assertLess(error, 1.0, "Quantization error should be less than 1.0")
        print(f"\n✓ SOM training working, quantization error: {error:.3f}")

        # 2. Test that different patterns map to different neurons
        neuron1 = self.som.winner(np.array([1.0, 1.1, 1.2]))  # Upward trend
        neuron2 = self.som.winner(np.array([1.0, 0.9, 0.8]))  # Downward trend
        self.assertNotEqual(neuron1, neuron2, "Different patterns should map to different neurons.")
        print("✓ SOM correctly differentiates distinct patterns.")

    def test_som_analysis(self):
        """
        Validation test for SOM pattern analysis.
        """
        neuron_map = analyze_som_patterns(self.som, self.patterns, self.future_returns)

        self.assertIsInstance(neuron_map, dict, "Analysis should return a dictionary.")
        self.assertGreater(len(neuron_map), 1, "Neuron map should contain multiple winning neurons.")

        # Check that some neurons have different average returns
        neuron_returns = list(neuron_map.values())
        differentiation = max(neuron_returns) - min(neuron_returns)

        # The max return should be positive (from upward patterns) and min negative
        self.assertGreater(max(neuron_returns), 0, "Max avg return should be positive.")
        self.assertLess(min(neuron_returns), 0, "Min avg return should be negative.")

        # The spread should be significant
        self.assertGreater(differentiation, 0.01, "Analysis should show differentiation between neurons.")
        print(f"✓ SOM analysis showing differentiation between neurons (spread: {differentiation:.4f})")

if __name__ == '__main__':
    unittest.main()
