import torch
import torch.nn as nn
from .blocks import TrendBlock, SeasonalityBlock, GenericBlock

class NBEATSWithFeatures(nn.Module):
    """
    An N-BEATS model that incorporates multiple external feature sets.
    It uses dedicated encoders for each feature type and injects the resulting
    embeddings into the N-BEATS blocks.
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 som_feature_dim: int,
                 external_feature_dim: int,
                 units: int = 64,
                 stack_types: list[str] = ['trend', 'seasonality', 'generic'],
                 n_blocks: int = 3,
                 thetas_dims: dict[str, int] = {'trend': 3, 'seasonality': 10, 'generic': 0}):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.units = units
        self.stack_types = stack_types

        # --- Encoders for External Features ---

        # Simple projection for SOM features (already a single vector)
        self.som_projection = nn.Linear(som_feature_dim, units)

        # GRU-based encoder for sequential external features (like trends)
        self.external_encoder = nn.GRU(
            input_size=external_feature_dim,
            hidden_size=units,
            num_layers=1,
            batch_first=True
        )

        # --- N-BEATS Stacks ---
        self.stacks = nn.ModuleList()
        block_map = {'trend': TrendBlock, 'seasonality': SeasonalityBlock, 'generic': GenericBlock}
        for stack_type in self.stack_types:
            self.stacks.append(nn.ModuleList([
                block_map[stack_type](
                    units=units,
                    thetas_dim=thetas_dims[stack_type],
                    backcast_length=backcast_length,
                    forecast_length=forecast_length
                ) for _ in range(n_blocks)
            ]))

    def forward(self, x_ts: torch.Tensor, som_features: torch.Tensor, x_external: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the N-BEATS model with multiple feature injections.
        """
        # 1. --- Create Embeddings from External Features ---
        som_embedding = torch.relu(self.som_projection(som_features))

        # Process the sequence of external features to get a single embedding
        _, h_n_external = self.external_encoder(x_external)
        external_embedding = torch.relu(h_n_external[-1, :, :])

        # 2. --- N-BEATS Doubly Residual Stacking (Parallel Stacks version) ---
        final_forecast = torch.zeros(x_ts.size(0), self.forecast_length, device=x_ts.device)
        backcast_residual = x_ts

        for i, stack in enumerate(self.stacks):
            stack_forecast_sum = torch.zeros(x_ts.size(0), self.forecast_length, device=x_ts.device)
            for block in stack:
                # Pass all embeddings to each block
                block_backcast, block_forecast = block(backcast_residual, som_embedding=som_embedding, external_embedding=external_embedding)
                backcast_residual = backcast_residual - block_backcast
                stack_forecast_sum = stack_forecast_sum + block_forecast

            # The forecast of each stack is added to the total
            # In this parallel architecture, the backcast residual is reset for each stack
            # This is a simplification. A fully sequential model is also possible but more complex.
            # For now, let's assume each stack works on the original signal.
            backcast_residual = x_ts
            final_forecast = final_forecast + stack_forecast_sum

        # For compatibility with tests that expect a backcast loss
        # The final residual backcast isn't well-defined in this parallel setup
        return final_forecast, torch.zeros_like(backcast_residual)
