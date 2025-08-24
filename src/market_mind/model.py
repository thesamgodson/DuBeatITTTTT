import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the Python path for sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from market_mind.participants import InformedTraderModule, NoiseTraderModule, MarketMakerModule, RegimeDetectionModule

class MarketMindModel(nn.Module):
    """
    The main model that aggregates signals from different participant modules
    using a regime-aware attention mechanism.
    """
    def __init__(self,
                 informed_trader_features: int,
                 noise_trader_features: int,
                 market_maker_features: int,
                 regime_features: int,
                 hidden_dim: int = 32,
                 num_attention_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. Instantiate all the sub-modules
        self.informed_trader_module = InformedTraderModule(informed_trader_features, hidden_dim)
        self.noise_trader_module = NoiseTraderModule(noise_trader_features, hidden_dim)
        self.market_maker_module = MarketMakerModule(market_maker_features, hidden_dim)
        self.regime_detection_module = RegimeDetectionModule(regime_features, hidden_dim)

        # 2. The attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True # Expects input as (batch, seq, features)
        )

        # 3. Final output layer
        self.output_layer = nn.Linear(hidden_dim, 1) # Predict a single value (the return)

    def forward(self,
                x_informed: torch.Tensor,
                x_noise: torch.Tensor,
                x_mm: torch.Tensor,
                x_regime: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the MarketMind model.

        Args:
            x_informed: Input tensor for the informed trader module.
            x_noise: Input tensor for the noise trader module.
            x_mm: Input tensor for the market maker module.
            x_regime: Input tensor for the regime detection module.

        Returns:
            torch.Tensor: The final prediction of shape [batch_size, 1].
        """
        # 1. Get embeddings from all modules
        informed_embedding = self.informed_trader_module(x_informed)
        noise_embedding = self.noise_trader_module(x_noise)
        market_maker_embedding = self.market_maker_module(x_mm)
        regime_embedding = self.regime_detection_module(x_regime)

        # 2. Prepare inputs for the attention mechanism
        # The participant embeddings are the source of information (keys and values)
        participant_embeddings = torch.stack(
            [informed_embedding, noise_embedding, market_maker_embedding],
            dim=1
        ) # Shape: [batch_size, 3, hidden_dim]

        # The regime embedding is the context that asks the question (query)
        # It needs a sequence dimension of 1
        query = regime_embedding.unsqueeze(1) # Shape: [batch_size, 1, hidden_dim]

        # 3. Apply attention
        # The attention layer will use the regime query to create a weighted average
        # of the participant embeddings.
        context_vector, _ = self.attention(
            query=query,
            key=participant_embeddings,
            value=participant_embeddings,
            need_weights=False # We don't need the attention weights for the forward pass
        )

        # The output context_vector has shape [batch_size, 1, hidden_dim],
        # so we squeeze out the sequence dimension.
        context_vector = context_vector.squeeze(1)

        # 4. Final prediction
        prediction = self.output_layer(context_vector)

        return prediction
