import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hypergraph_convolution.layers import CustomHypergraphConv

class TestPhase3HypergraphConvolution(unittest.TestCase):

    def setUp(self):
        """Set up common variables for tests."""
        self.num_nodes = 5
        self.in_channels = 8
        self.out_channels = 16

        # A simple hypergraph with 2 hyperedges: {0, 1, 2} and {2, 3, 4}
        self.hyperedge_index = torch.tensor([
            [0, 1, 2, 2, 3, 4],  # Node indices
            [0, 0, 0, 1, 1, 1]   # Hyperedge IDs
        ], dtype=torch.long)

        self.x = torch.randn(self.num_nodes, self.in_channels)

    def test_layer_output_shape(self):
        """
        Tests that the layer produces an output with the correct shape.
        """
        print("\n\n--- Testing Hypergraph Layer Output Shape ---")
        layer = CustomHypergraphConv(self.in_channels, self.out_channels)
        output = layer(self.x, self.hyperedge_index)

        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        print("✓ Output shape is correct.")

    def test_layer_is_trainable(self):
        """
        Tests that the layer's parameters have gradients after a backward pass.
        """
        print("\n--- Testing if Hypergraph Layer is Trainable ---")
        layer = CustomHypergraphConv(self.in_channels, self.out_channels)

        # Ensure parameters require gradients
        self.assertTrue(layer.theta.weight.requires_grad)

        # Perform a forward and backward pass
        output = layer(self.x, self.hyperedge_index)
        output.sum().backward()

        # Check that the gradients are not None
        self.assertIsNotNone(layer.theta.weight.grad, "Theta weights have no gradient.")
        self.assertIsNotNone(layer.hyperedge_weight_param.grad, "Hyperedge weights have no gradient.")
        print("✓ Gradients are computed correctly, layer is trainable.")

    def test_no_hyperedges_case(self):
        """
        Tests the layer's behavior when there are no hyperedges.
        """
        print("\n--- Testing No Hyperedges Case ---")
        empty_hyperedge_index = torch.empty((2, 0), dtype=torch.long)
        layer = CustomHypergraphConv(self.in_channels, self.out_channels)

        output = layer(self.x, empty_hyperedge_index)

        # The output should just be the result of the linear transformation
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

        # Check that it's not just returning the input
        # This is an indirect way to check if the linear layer was applied.
        # A more direct check would be to compare with a manual linear layer application.
        manual_output = layer.theta(self.x)
        self.assertTrue(torch.allclose(output, manual_output))
        print("✓ Correctly handles input with no hyperedges.")

if __name__ == '__main__':
    unittest.main()
