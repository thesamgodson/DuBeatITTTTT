import numpy as np
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Add the src directory to the Python path to allow sibling imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.phase_1 import PatternExtractor

class HypergraphBuilder:
    """
    Constructs a hypergraph from a time series based on pattern similarity using
    the clique expansion method.
    """
    def __init__(self,
                 pattern_window_size: int = 7,
                 similarity_threshold: float = 0.95):
        """
        Args:
            pattern_window_size (int): The size of the sliding window to create patterns.
            similarity_threshold (float): The cosine similarity threshold to connect patterns.
        """
        self.pattern_window_size = pattern_window_size
        self.similarity_threshold = similarity_threshold
        self.pattern_extractor = PatternExtractor(windows=[self.pattern_window_size])

    def build(self, price_sequence: np.ndarray, asset_name: str = "asset") -> torch.Tensor:
        """
        Builds a hypergraph and returns its hyperedge index representation.

        Args:
            price_sequence (np.ndarray): The 1D array of prices.
            asset_name (str): The name of the asset for logging purposes.

        Returns:
            torch.Tensor: The hyperedge index tensor of shape [2, num_nodes_in_all_hyperedges].
                          Returns an empty tensor if no hyperedges can be formed.
        """
        if len(price_sequence) < self.pattern_window_size:
            # Not enough data to build a meaningful graph
            return torch.empty((2, 0), dtype=torch.long)

        # 1. Create normalized patterns from the price sequence
        patterns_dict = self.pattern_extractor.extract_all_patterns(price_sequence)
        patterns = patterns_dict[self.pattern_window_size]

        if len(patterns) <= 1:
            # Not enough patterns to form edges
            return torch.empty((2, 0), dtype=torch.long)

        # 2. Calculate pairwise similarity between all patterns
        # We add a small epsilon to avoid issues with zero-variance patterns
        sim_matrix = cosine_similarity(patterns + 1e-9)

        # 3. Build a simple graph based on the similarity threshold
        # The nodes in this graph correspond to the start time of each pattern.
        num_nodes = len(patterns)
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))

        # Find pairs with similarity above the threshold
        # np.triu with k=1 ensures we only check the upper triangle (no self-loops, no duplicates)
        similar_pairs = np.argwhere(np.triu(sim_matrix, k=1) > self.similarity_threshold)
        graph.add_edges_from(similar_pairs)

        # 4. Find all maximal cliques in the graph. Each clique is a hyperedge.
        # A hyperedge must connect at least 2 nodes.
        cliques = [clique for clique in nx.find_cliques(graph) if len(clique) > 1]

        # 5. Convert the list of cliques into the torch_geometric hyperedge_index format
        if not cliques:
            return torch.empty((2, 0), dtype=torch.long)

        node_indices = []
        hyperedge_ids = []
        for i, clique in enumerate(cliques):
            node_indices.extend(clique)
            hyperedge_ids.extend([i] * len(clique))

        hyperedge_index = torch.tensor([node_indices, hyperedge_ids], dtype=torch.long)

        return hyperedge_index
