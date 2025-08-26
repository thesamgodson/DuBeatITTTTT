import torch
import torch.nn as nn
from typing import Tuple

class NBEATSBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, basis_function: nn.Module, layers: int, layer_size: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)] +
                                    [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])
        self.basis_function = basis_function
        self.theta_layer = nn.Linear(layer_size, theta_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        theta = self.theta_layer(block_input)
        backcast, forecast = self.basis_function(theta)
        return backcast, forecast

class NBEATS(nn.Module):
    def __init__(self,
                 input_size: int,
                 forecast_horizon: int,
                 n_stacks: int,
                 n_blocks: int,
                 n_layers: int,
                 layer_size: int,
                 theta_size: int,
                 basis_function_class: nn.Module):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_size = input_size
        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            blocks = nn.ModuleList()
            for __ in range(n_blocks):
                basis_function = basis_function_class(self.input_size, self.forecast_horizon, theta_size)
                block = NBEATSBlock(self.input_size, theta_size, basis_function, n_layers, layer_size)
                blocks.append(block)
            self.stacks.append(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = x.clone()
        forecast = torch.zeros(x.size(0), self.forecast_horizon).to(x.device)
        for stack in self.stacks:
            stack_forecast = torch.zeros(x.size(0), self.forecast_horizon).to(x.device)
            for block in stack:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                stack_forecast = stack_forecast + block_forecast
            forecast = forecast + stack_forecast
        return forecast

class TrendBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size, theta_size):
        super().__init__()
        self.p = theta_size
        self.T = torch.arange(backcast_size + forecast_size, dtype=torch.float32)
        self.T_backcast = self.T[:backcast_size]
        self.T_forecast = self.T[-forecast_size:]

    def forward(self, theta):
        # Ensure T_backcast and T_forecast are on the same device as theta
        T_backcast = self.T_backcast.to(theta.device)
        T_forecast = self.T_forecast.to(theta.device)

        # Create polynomial terms
        powers = torch.arange(self.p, dtype=torch.float32).to(theta.device)
        backcast_T = T_backcast.unsqueeze(0) ** powers.unsqueeze(1)
        forecast_T = T_forecast.unsqueeze(0) ** powers.unsqueeze(1)

        backcast = torch.einsum('bp,pt->bt', theta, backcast_T)
        forecast = torch.einsum('bp,pt->bt', theta, forecast_T)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size, theta_size):
        super().__init__()
        self.p = theta_size
        self.num_harmonics = self.p // 2

        # Frequencies for Fourier series
        frequencies = torch.arange(1, self.num_harmonics + 1, dtype=torch.float32) * 2 * torch.pi

        # Time vectors
        T_backcast = torch.arange(backcast_size, dtype=torch.float32) / backcast_size
        T_forecast = torch.arange(backcast_size, backcast_size + forecast_size, dtype=torch.float32) / (backcast_size + forecast_size)

        self.backcast_cos = torch.cos(frequencies.unsqueeze(1) * T_backcast.unsqueeze(0))
        self.backcast_sin = torch.sin(frequencies.unsqueeze(1) * T_backcast.unsqueeze(0))
        self.forecast_cos = torch.cos(frequencies.unsqueeze(1) * T_forecast.unsqueeze(0))
        self.forecast_sin = torch.sin(frequencies.unsqueeze(1) * T_forecast.unsqueeze(0))

    def forward(self, theta):
        # Ensure basis functions are on the same device as theta
        backcast_cos = self.backcast_cos.to(theta.device)
        backcast_sin = self.backcast_sin.to(theta.device)
        forecast_cos = self.forecast_cos.to(theta.device)
        forecast_sin = self.forecast_sin.to(theta.device)

        # Split theta for cos and sin coefficients
        theta_cos = theta[:, :self.num_harmonics]
        theta_sin = theta[:, self.num_harmonics:]

        backcast = torch.einsum('bp,pt->bt', theta_cos, backcast_cos) + torch.einsum('bp,pt->bt', theta_sin, backcast_sin)
        forecast = torch.einsum('bp,pt->bt', theta_cos, forecast_cos) + torch.einsum('bp,pt->bt', theta_sin, forecast_sin)
        return backcast, forecast


class NBEATSWithSOM(NBEATS):
    def __init__(self, som_feature_extractor, sequence_length, forecast_horizon, n_stacks, n_blocks, n_layers, layer_width):
        self.som_feature_extractor = som_feature_extractor

        # The NBEATS generic model will be used for the main forecasting part.
        # We define its parameters here.
        # The theta_size for the TrendBasis is the degree of the polynomial.
        # A small degree like 2 or 3 is usually sufficient.
        trend_theta_size = 3

        # The theta_size for SeasonalityBasis is 2 * number of harmonics.
        # Let's choose a reasonable number of harmonics.
        seasonality_theta_size = 20

        # We will create two stacks: one for trend, one for seasonality.
        if n_stacks != 2:
            print("Warning: NBEATSWithSOM is designed for 2 stacks (trend, seasonality). Overriding n_stacks to 2.")
            n_stacks = 2

        # Create the NBEATS model with two stacks
        super().__init__(
            input_size=sequence_length, # The input to NBEATS is the time series sequence
            forecast_horizon=forecast_horizon,
            n_stacks=0, # Stacks will be added manually
            n_blocks=n_blocks,
            n_layers=n_layers,
            layer_size=layer_width,
            theta_size=0, # Will be set per stack
            basis_function_class=None # Will be set per stack
        )

        # Manually create stacks
        self.stacks = nn.ModuleList()

        # Trend Stack
        trend_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            basis = TrendBasis(backcast_size=sequence_length, forecast_size=forecast_horizon, theta_size=trend_theta_size)
            block = NBEATSBlock(sequence_length, trend_theta_size, basis, n_layers, layer_width)
            trend_blocks.append(block)
        self.stacks.append(trend_blocks)

        # Seasonality Stack
        seasonality_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            basis = SeasonalityBasis(backcast_size=sequence_length, forecast_size=forecast_horizon, theta_size=seasonality_theta_size)
            block = NBEATSBlock(sequence_length, seasonality_theta_size, basis, n_layers, layer_width)
            seasonality_blocks.append(block)
        self.stacks.append(seasonality_blocks)

        # Gating mechanism based on SOM features
        dummy_input = torch.randn(1, sequence_length)
        som_feature_size = self.som_feature_extractor.extract_features(dummy_input).shape[1]
        self.gate_layer = nn.Linear(som_feature_size, n_stacks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract SOM features
        som_features = self.som_feature_extractor.extract_features(x)

        # Calculate gate weights (softmax to ensure they sum to 1)
        gate_weights = torch.softmax(self.gate_layer(som_features), dim=1)

        # Process each stack and combine forecasts
        residuals = x.clone()
        total_forecast = torch.zeros(x.size(0), self.forecast_horizon).to(x.device)

        for i, stack in enumerate(self.stacks):
            stack_forecast = torch.zeros(x.size(0), self.forecast_horizon).to(x.device)
            for block in stack:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                stack_forecast = stack_forecast + block_forecast

            # Apply gating
            total_forecast += stack_forecast * gate_weights[:, i].unsqueeze(1)

        return total_forecast
