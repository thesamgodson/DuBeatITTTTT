import torch
import torch.nn as nn
from .blocks import TrendBlock, SeasonalityBlock, GenericBlock

class NBEATSWithSOM(nn.Module):
    """
    An N-BEATS model that incorporates SOM features, with regularization.
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 som_feature_dim: int,
                 units: int = 64,
                 stack_types: list[str] = ['trend', 'seasonality', 'generic'],
                 n_blocks: int = 3,
                 thetas_dims: dict[str, int] = {'trend': 3, 'seasonality': 10, 'generic': 0}):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.som_feature_dim = som_feature_dim
        self.units = units
        self.stack_types = stack_types

        # Stack-specific projection layers for SOM features
        self.som_projections = nn.ModuleDict()
        for stack_type in self.stack_types:
            self.som_projections[stack_type] = nn.Linear(som_feature_dim, units)

        # Learnable weights for each stack's SOM embedding (regularization)
        self.stack_som_weights = nn.ParameterDict()
        for stack_type in self.stack_types:
            self.stack_som_weights[stack_type] = nn.Parameter(torch.tensor(0.1))

        # N-BEATS stacks
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

    def forward(self, x_ts: torch.Tensor, som_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the N-BEATS model with SOM feature injection.
        """
        final_forecast = torch.zeros(x_ts.size(0), self.forecast_length, device=x_ts.device)
        backcast_residual = x_ts

        for i, stack_type in enumerate(self.stack_types):
            stack = self.stacks[i]

            # Create stack-specific embedding
            projection_layer = self.som_projections[stack_type]
            som_embedding_raw = torch.relu(projection_layer(som_features))

            # Apply learnable weight
            stack_weight = self.stack_som_weights[stack_type]
            som_embedding = stack_weight * som_embedding_raw

            stack_forecast_sum = torch.zeros(x_ts.size(0), self.forecast_length, device=x_ts.device)
            for block in stack:
                block_backcast, block_forecast = block(backcast_residual, som_embedding=som_embedding)
                backcast_residual = backcast_residual - block_backcast
                stack_forecast_sum = stack_forecast_sum + block_forecast

            final_forecast = final_forecast + stack_forecast_sum

        # For compatibility with tests that expect a backcast loss
        return final_forecast, backcast_residual
