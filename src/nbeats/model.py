import torch
import torch.nn as nn
from .blocks import TrendBlock, SeasonalityBlock, GenericBlock

class PureNBEATS(nn.Module):
    """
    A pure N-BEATS model implementation without any topological features.
    The model is composed of stacks of trend, seasonality, and generic blocks.
    """
    def __init__(self,
                 backcast_length: int,
                 forecast_length: int,
                 stack_types: list[str] = ['trend', 'seasonality', 'generic'],
                 n_blocks: int = 3,
                 units: int = 64,
                 thetas_dims: dict[str, int] = {'trend': 3, 'seasonality': 10, 'generic': 0}):
        """
        Args:
            backcast_length (int): The length of the lookback window.
            forecast_length (int): The length of the forecast horizon.
            stack_types (list[str]): A list of stack types to include, e.g., ['trend', 'seasonality'].
            n_blocks (int): The number of blocks per stack.
            units (int): The number of units in the fully connected layers of each block.
            thetas_dims (dict[str, int]): A dictionary specifying the thetas_dim for each block type.
        """
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stacks = nn.ModuleList()

        block_map = {
            'trend': TrendBlock,
            'seasonality': SeasonalityBlock,
            'generic': GenericBlock
        }

        for stack_type in stack_types:
            if stack_type not in block_map:
                raise ValueError(f"Unknown stack type: {stack_type}")

            stack_blocks = []
            for _ in range(n_blocks):
                block = block_map[stack_type](
                    units=units,
                    thetas_dim=thetas_dims[stack_type],
                    backcast_length=backcast_length,
                    forecast_length=forecast_length
                )
                stack_blocks.append(block)

            self.stacks.append(nn.ModuleList(stack_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass with doubly residual stacking.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, backcast_length).

        Returns:
            torch.Tensor: The final forecast tensor of shape (batch_size, forecast_length).
        """
        # Ensure input is of the correct shape
        if x.shape[1] != self.backcast_length:
            raise ValueError(f"Input tensor shape[1] {x.shape[1]} does not match backcast_length {self.backcast_length}")

        # Initialize residual backcast and final forecast
        residual_backcast = x
        final_forecast = torch.zeros(x.size(0), self.forecast_length, device=x.device)

        # Iterate through each stack and each block within the stack
        for stack in self.stacks:
            for block in stack:
                # Get the block's backcast and forecast
                block_backcast, block_forecast = block(residual_backcast)

                # Update the residual backcast by subtracting the block's contribution
                residual_backcast = residual_backcast - block_backcast

                # Add the block's forecast to the final forecast
                final_forecast = final_forecast + block_forecast

        return final_forecast
