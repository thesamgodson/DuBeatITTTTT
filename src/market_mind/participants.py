import torch
import torch.nn as nn

class InformedTraderModule(nn.Module):
    """
    A module designed to model the behavior of informed traders.
    It processes features that suggest informed activity, like volume-weighted returns.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: An embedding of shape [batch_size, hidden_dim].
        """
        _, h_n = self.gru(x)
        last_hidden_state = h_n[-1, :, :]
        output = torch.tanh(self.output_layer(last_hidden_state))
        return output

class NoiseTraderModule(nn.Module):
    """
    A module designed to model the behavior of noise traders.
    It processes features that suggest noise or sentiment-driven activity.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: An embedding of shape [batch_size, hidden_dim].
        """
        _, h_n = self.gru(x)
        last_hidden_state = h_n[-1, :, :]
        output = torch.tanh(self.output_layer(last_hidden_state))
        return output

class MarketMakerModule(nn.Module):
    """
    A module designed to model the behavior of market makers.
    It processes features related to market liquidity and spreads.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: An embedding of shape [batch_size, hidden_dim].
        """
        _, h_n = self.gru(x)
        last_hidden_state = h_n[-1, :, :]
        output = torch.tanh(self.output_layer(last_hidden_state))
        return output

class RegimeDetectionModule(nn.Module):
    """
    A module to identify the current market regime (e.g., volatile, quiet).
    It processes market-wide features like volatility and ATR.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: A regime embedding of shape [batch_size, hidden_dim].
        """
        _, h_n = self.gru(x)
        last_hidden_state = h_n[-1, :, :]
        output = torch.tanh(self.output_layer(last_hidden_state))
        return output
