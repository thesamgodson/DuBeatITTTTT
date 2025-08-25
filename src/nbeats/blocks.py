import torch
import torch.nn as nn

class TrendBlock(nn.Module):
    def __init__(self, units: int, thetas_dim: int, backcast_length: int, forecast_length: int):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)
        t_backcast = torch.arange(backcast_length, dtype=torch.float32) / backcast_length
        t_forecast = torch.arange(forecast_length, dtype=torch.float32) / forecast_length
        self.p_backcast = torch.stack([t_backcast ** i for i in range(thetas_dim)], dim=0)
        self.p_forecast = torch.stack([t_forecast ** i for i in range(thetas_dim)], dim=0)

    def forward(self, x: torch.Tensor, som_embedding: torch.Tensor = None, external_embedding: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        if som_embedding is not None:
            x = x + som_embedding
        if external_embedding is not None:
            x = x + external_embedding
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        backcast = torch.einsum('bp,pt->bt', theta_b, self.p_backcast)
        forecast = torch.einsum('bp,pt->bt', theta_f, self.p_forecast)
        return backcast, forecast

class SeasonalityBlock(nn.Module):
    def __init__(self, units: int, thetas_dim: int, backcast_length: int, forecast_length: int):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.n_harmonics = thetas_dim // 2
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)
        t_backcast = torch.arange(backcast_length, dtype=torch.float32) / backcast_length
        t_forecast = torch.arange(forecast_length, dtype=torch.float32) / forecast_length
        harmonics = torch.arange(1.0, self.n_harmonics + 1)
        s_backcast_cos = torch.cos(2 * torch.pi * harmonics[:, None] * t_backcast[None, :])
        s_backcast_sin = torch.sin(2 * torch.pi * harmonics[:, None] * t_backcast[None, :])
        self.s_backcast = torch.cat([s_backcast_cos, s_backcast_sin], dim=0)
        s_forecast_cos = torch.cos(2 * torch.pi * harmonics[:, None] * t_forecast[None, :])
        s_forecast_sin = torch.sin(2 * torch.pi * harmonics[:, None] * t_forecast[None, :])
        self.s_forecast = torch.cat([s_forecast_cos, s_forecast_sin], dim=0)

    def forward(self, x: torch.Tensor, som_embedding: torch.Tensor = None, external_embedding: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        if som_embedding is not None:
            x = x + som_embedding
        if external_embedding is not None:
            x = x + external_embedding
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        backcast = torch.einsum('bp,pt->bt', theta_b, self.s_backcast)
        forecast = torch.einsum('bp,pt->bt', theta_f, self.s_forecast)
        return backcast, forecast

class GenericBlock(nn.Module):
    def __init__(self, units: int, thetas_dim: int, backcast_length: int, forecast_length: int):
        super().__init__()
        self.units = units
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.backcast_layer = nn.Linear(units, backcast_length)
        self.forecast_layer = nn.Linear(units, forecast_length)

    def forward(self, x: torch.Tensor, som_embedding: torch.Tensor = None, external_embedding: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        if som_embedding is not None:
            x = x + som_embedding
        if external_embedding is not None:
            x = x + external_embedding
        backcast = self.backcast_layer(x)
        forecast = self.forecast_layer(x)
        return backcast, forecast
