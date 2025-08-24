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
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: An embedding of shape [batch_size, 1].
        """
        # The GRU returns the output of all timesteps and the final hidden state.
        # We only need the final hidden state.
        _, h_n = self.gru(x)

        # h_n shape is [num_layers, batch_size, hidden_dim]. We take the last layer's state.
        last_hidden_state = h_n[-1, :, :]

        # Pass the final hidden state through a linear layer to get the output embedding
        output = self.output_layer(last_hidden_state)
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
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: An embedding of shape [batch_size, 1].
        """
        _, h_n = self.gru(x)
        last_hidden_state = h_n[-1, :, :]
        output = self.output_layer(last_hidden_state)
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
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].

        Returns:
            torch.Tensor: An embedding of shape [batch_size, 1].
        """
        _, h_n = self.gru(x)
        last_hidden_state = h_n[-1, :, :]
        output = self.output_layer(last_hidden_state)
        return output
