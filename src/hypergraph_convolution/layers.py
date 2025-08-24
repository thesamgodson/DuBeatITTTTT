import torch
import torch.nn as nn

class CustomHypergraphConv(nn.Module):
    """
    A custom implementation of a Hypergraph Convolutional Layer.
    This layer performs message passing on a hypergraph structure, updating node
    features based on the higher-order relationships defined by hyperedges.

    The operation is based on the formula from the "Hypergraph Convolution and
    Hypergraph Attention" paper: X' = D⁻¹ * H * W * B⁻¹ * Hᵀ * X * Θ
    """
    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = True):
        """
        Args:
            in_channels (int): Number of features for each input node.
            out_channels (int): Number of features for each output node.
            use_bias (bool): Whether to use a bias term in the linear transformation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Θ: The learnable linear transformation applied to node features.
        self.theta = nn.Linear(in_channels, out_channels, bias=use_bias)

        # W: A learnable weight for each hyperedge. This will be dynamically sized.
        # We don't initialize it here because the number of hyperedges is not known
        # until the forward pass. We will create it on the fly.
        self.hyperedge_weight_param = None

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the hypergraph convolution.

        Args:
            x (torch.Tensor): Input node features. Shape: [num_nodes, in_channels].
            hyperedge_index (torch.Tensor): The hypergraph structure.
                                          Shape: [2, num_total_hyperedge_nodes].

        Returns:
            torch.Tensor: Updated node features. Shape: [num_nodes, out_channels].
        """
        num_nodes = x.shape[0]
        if hyperedge_index.shape[1] == 0:
            # If there are no hyperedges, just apply the linear transformation
            return self.theta(x)

        num_hyperedges = int(hyperedge_index[1].max()) + 1

        # Lazily initialize the hyperedge weights parameter if it doesn't exist
        # or if the number of hyperedges has changed.
        if self.hyperedge_weight_param is None or self.hyperedge_weight_param.shape[0] != num_hyperedges:
            # W: A learnable weight for each hyperedge.
            self.hyperedge_weight_param = nn.Parameter(torch.ones(num_hyperedges, device=x.device))

        # First, apply the linear transformation Θ to the node features X.
        x_transformed = self.theta(x)

        # Build the sparse incidence matrix H from the hyperedge_index
        # H has shape [num_nodes, num_hyperedges]
        H = torch.sparse_coo_tensor(
            indices=hyperedge_index,
            values=torch.ones(hyperedge_index.shape[1], device=x.device),
            size=(num_nodes, num_hyperedges)
        )

        # --- Compute degrees ---
        # B: Hyperedge degrees (number of nodes per hyperedge).
        # We add 1e-6 to avoid division by zero for isolated hyperedges.
        B_inv_diag = 1.0 / (torch.sparse.sum(H, dim=0).to_dense() + 1e-6)

        # D: Node degrees (number of hyperedges per node).
        D_inv_diag = 1.0 / (torch.sparse.sum(H, dim=1).to_dense() + 1e-6)

        # --- Perform message passing ---

        # Step 1: Aggregate node features into hyperedge features (Hᵀ * X * Θ)
        # Result shape: [num_hyperedges, out_channels]
        hyperedge_features = torch.sparse.mm(H.t(), x_transformed)

        # Step 2: Apply hyperedge degree normalization (B⁻¹ * Hᵀ * X * Θ)
        hyperedge_features = hyperedge_features * B_inv_diag.unsqueeze(1)

        # Step 3: Apply learnable hyperedge weights (W * B⁻¹ * Hᵀ * X * Θ)
        hyperedge_features = hyperedge_features * self.hyperedge_weight_param.unsqueeze(1)

        # Step 4: Propagate hyperedge features back to nodes (H * W * B⁻¹ * Hᵀ * X * Θ)
        # Result shape: [num_nodes, out_channels]
        node_features_aggregated = torch.sparse.mm(H, hyperedge_features)

        # Step 5: Apply node degree normalization (D⁻¹ * H * W * B⁻¹ * Hᵀ * X * Θ)
        output = node_features_aggregated * D_inv_diag.unsqueeze(1)

        return output
