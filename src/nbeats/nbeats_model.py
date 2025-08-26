import torch
import torch.nn as nn
from typing import Tuple, Optional

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

class TrendBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size, theta_size):
        super().__init__()
        self.p = theta_size
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        powers = torch.arange(self.p, device=theta.device, dtype=torch.float32)

        t_back = torch.arange(self.backcast_size, device=theta.device, dtype=torch.float32)
        backcast_basis = t_back.unsqueeze(0).pow(powers.unsqueeze(1))
        backcast = torch.einsum('bp,pt->bt', theta, backcast_basis)

        t_fore = torch.arange(self.forecast_size, device=theta.device, dtype=torch.float32)
        forecast_basis = t_fore.unsqueeze(0).pow(powers.unsqueeze(1))
        forecast = torch.einsum('bp,pt->bt', theta, forecast_basis)

        return backcast, forecast

class SeasonalityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size, theta_size):
        super().__init__()
        self.num_harmonics = theta_size // 2
        frequencies = torch.arange(1, self.num_harmonics + 1, dtype=torch.float32) * torch.pi

        t_back = torch.arange(backcast_size, dtype=torch.float32)
        self.backcast_cos = torch.cos(frequencies.unsqueeze(1) * t_back.unsqueeze(0))
        self.backcast_sin = torch.sin(frequencies.unsqueeze(1) * t_back.unsqueeze(0))

        t_fore = torch.arange(forecast_size, dtype=torch.float32)
        self.forecast_cos = torch.cos(frequencies.unsqueeze(1) * t_fore.unsqueeze(0))
        self.forecast_sin = torch.sin(frequencies.unsqueeze(1) * t_fore.unsqueeze(0))

    def forward(self, theta):
        device = theta.device
        theta_cos = theta[:, :self.num_harmonics]
        theta_sin = theta[:, self.num_harmonics:]

        backcast = torch.einsum('bp,pt->bt', theta_cos, self.backcast_cos.to(device)) + \
                   torch.einsum('bp,pt->bt', theta_sin, self.backcast_sin.to(device))

        forecast = torch.einsum('bp,pt->bt', theta_cos, self.forecast_cos.to(device)) + \
                   torch.einsum('bp,pt->bt', theta_sin, self.forecast_sin.to(device))

        return backcast, forecast

class NBEATSWithSOM(nn.Module):
    def __init__(self,
                 sequence_length: int,
                 forecast_horizon: int,
                 n_stacks: int,
                 n_blocks: int,
                 n_layers: int,
                 layer_width: int,
                 num_features: int = 1,
                 som_feature_extractor: Optional[nn.Module] = None,
                 use_som_gating: bool = True):
        super().__init__()

        self.use_som_gating = use_som_gating and (som_feature_extractor is not None)
        self.som_feature_extractor = som_feature_extractor
        self.forecast_horizon = forecast_horizon

        nbeats_input_size = sequence_length * num_features
        trend_theta_size = 4
        seasonality_theta_size = 20

        if n_stacks != 2:
            print("Warning: NBEATS model is designed for 2 stacks (trend, seasonality). Overriding n_stacks to 2.")
            n_stacks = 2

        self.stacks = nn.ModuleList()

        # Trend Stack
        trend_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            basis = TrendBasis(backcast_size=nbeats_input_size, forecast_size=forecast_horizon, theta_size=trend_theta_size)
            block = NBEATSBlock(nbeats_input_size, trend_theta_size, basis, n_layers, layer_width)
            trend_blocks.append(block)
        self.stacks.append(trend_blocks)

        # Seasonality Stack
        seasonality_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            basis = SeasonalityBasis(backcast_size=nbeats_input_size, forecast_size=forecast_horizon, theta_size=seasonality_theta_size)
            block = NBEATSBlock(nbeats_input_size, seasonality_theta_size, basis, n_layers, layer_width)
            seasonality_blocks.append(block)
        self.stacks.append(seasonality_blocks)

        if self.use_som_gating:
            dummy_som_input = torch.randn(1, sequence_length)
            som_feature_size = self.som_feature_extractor.extract_features(dummy_som_input).shape[1]
            self.gate_layer = nn.Linear(som_feature_size, n_stacks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        flat_x = x.view(x.size(0), -1)
        residuals = flat_x.clone()
        total_forecast = torch.zeros(x.size(0), self.forecast_horizon).to(x.device)

        som_features = None
        if self.use_som_gating:
            close_price_sequence = x[:, :, 0]
            som_features = self.som_feature_extractor.extract_features(close_price_sequence)
            gate_weights = torch.softmax(self.gate_layer(som_features), dim=1)

        for i, stack in enumerate(self.stacks):
            stack_forecast = torch.zeros(x.size(0), self.forecast_horizon).to(x.device)
            for block in stack:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                stack_forecast = stack_forecast + block_forecast

            if self.use_som_gating:
                total_forecast += stack_forecast * gate_weights[:, i].unsqueeze(1)
            else:
                total_forecast += stack_forecast

        return total_forecast, som_features
