import unittest
import numpy as np
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hypergraph_builder.builder import HypergraphBuilder

class TestPhase2HypergraphBuilder(unittest.TestCase):

    def test_basic_hypergraph_creation(self):
        """
        Tests that a hypergraph is created correctly for a sequence with similar patterns.
        """
        print("\n\n--- Testing Basic Hypergraph Creation ---")
        # A sequence where the pattern [1, 2, 3] repeats
        price_sequence = np.array([1, 2, 3, 4, 5, 1, 2, 3, 6, 7], dtype=np.float32)

        # We expect the patterns starting at index 0 and index 5 to be highly similar
        builder = HypergraphBuilder(pattern_window_size=3, similarity_threshold=0.99)
        hyperedge_index = builder.build(price_sequence)

        # 1. Validate the output type and shape
        self.assertIsInstance(hyperedge_index, torch.Tensor)
        self.assertEqual(hyperedge_index.dtype, torch.long)
        self.assertEqual(hyperedge_index.shape[0], 2)

        # 2. Validate that a hyperedge was created
        self.assertGreater(hyperedge_index.shape[1], 0, "No hyperedges were created.")

        # 3. Validate the content of the hyperedge
        # The first pattern (node 0) and the sixth pattern (node 5) should be in a clique.
        nodes = hyperedge_index[0].numpy()
        edges = hyperedge_index[1].numpy()

        # Find the hyperedge ID for node 0
        edge_id_for_node_0 = edges[np.where(nodes == 0)[0][0]]

        # Find all nodes belonging to that same hyperedge
        nodes_in_same_edge = nodes[np.where(edges == edge_id_for_node_0)[0]]

        self.assertIn(0, nodes_in_same_edge)
        self.assertIn(5, nodes_in_same_edge)
        print("✓ Correctly identified similar patterns and created a hyperedge.")

    def test_no_hyperedges_created(self):
        """
        Tests that no hyperedges are created when no patterns are similar.
        """
        print("\n--- Testing No Hyperedge Creation ---")
        # Use an impossibly high threshold to guarantee no edges are formed.
        price_sequence = np.random.rand(50).astype(np.float32)

        builder = HypergraphBuilder(pattern_window_size=5, similarity_threshold=1.01)
        hyperedge_index = builder.build(price_sequence)

        self.assertEqual(hyperedge_index.shape[1], 0, "Should not have created any hyperedges with a threshold > 1.0.")
        print("✓ Correctly handled sequence with no similar patterns.")

    def test_insufficient_data_handling(self):
        """
        Tests that the builder handles sequences shorter than the window size gracefully.
        """
        print("\n--- Testing Insufficient Data Handling ---")
        price_sequence = np.array([1, 2, 3], dtype=np.float32)

        builder = HypergraphBuilder(pattern_window_size=5, similarity_threshold=0.95)
        hyperedge_index = builder.build(price_sequence)

        self.assertIsInstance(hyperedge_index, torch.Tensor)
        self.assertEqual(hyperedge_index.shape[1], 0)
        print("✓ Correctly handled insufficient data.")

if __name__ == '__main__':
    unittest.main()
