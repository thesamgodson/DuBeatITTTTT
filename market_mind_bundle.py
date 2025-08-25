import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import unittest
import yfinance as yf
from typing import List, Dict, Tuple

# =============================================================================
# All source code from the project, consolidated into one file.
# =============================================================================

# --- From: src/data_processing/feature_engineering.py ---

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Returns'] = df['Close'].pct_change()
    df['LogReturns'] = np.log(df['Close']).diff()
    return df

def add_volatility_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['Volatility'] = df['LogReturns'].rolling(window=window).std()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=window).mean()
    return df

def add_microstructure_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['LiquidityProxy'] = (df['High'] - df['Low']) / df['Close']
    df['InformedTradingProxy'] = df['LogReturns'].abs() * df['Volume']
    df['NoiseTradingProxy'] = df['Volume'].rolling(window=window).corr(df['LogReturns'].abs()).fillna(0)
    return df

def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_microstructure_features(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- From: src/data_processing/pipeline.py ---

class MarketDataReader:
    def __init__(self, assets: List[str], start_date: str, end_date: str):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date

    def load_and_process(self) -> Dict[str, pd.DataFrame]:
        output_data = {}
        for asset in self.assets:
            try:
                raw_df = yf.download(asset, start=self.start_date, end=self.end_date, auto_adjust=True)
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                if raw_df.empty:
                    continue

                # Use the new variable name here
                processed_df = process_market_data(raw_df)
                output_data[asset] = processed_df
            except Exception as e:
                print(f"Error processing {asset}: {e}")
        return output_data

class Aligner:
    def __init__(self, backcast_length: int, forecast_horizon: int):
        self.backcast_length = backcast_length
        self.forecast_horizon = forecast_horizon

    def create_sequences(self, df: pd.DataFrame, target_column: str = 'LogReturns') -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        target_array = df[target_column].values
        feature_columns = [col for col in df.columns if col not in ['Date', target_column]]
        feature_array = df[feature_columns].values
        for i in range(self.backcast_length, len(df) - self.forecast_horizon + 1):
            X.append(feature_array[i - self.backcast_length : i])
            y.append(target_array[i : i + self.forecast_horizon])
        return np.array(X), np.array(y)

# --- From: src/market_mind/participants.py ---

class InformedTraderModule(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return torch.tanh(self.output_layer(h_n[-1, :, :]))

class NoiseTraderModule(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return torch.tanh(self.output_layer(h_n[-1, :, :]))

class MarketMakerModule(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return torch.tanh(self.output_layer(h_n[-1, :, :]))

class RegimeDetectionModule(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return torch.tanh(self.output_layer(h_n[-1, :, :]))

# --- From: src/market_mind/model.py ---

class MarketMindModel(nn.Module):
    def __init__(self, informed_trader_features: int, noise_trader_features: int, market_maker_features: int, regime_features: int, hidden_dim: int = 32, num_attention_heads: int = 4):
        super().__init__()
        self.informed_trader_module = InformedTraderModule(informed_trader_features, hidden_dim)
        self.noise_trader_module = NoiseTraderModule(noise_trader_features, hidden_dim)
        self.market_maker_module = MarketMakerModule(market_maker_features, hidden_dim)
        self.regime_detection_module = RegimeDetectionModule(regime_features, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_attention_heads, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x_informed: torch.Tensor, x_noise: torch.Tensor, x_mm: torch.Tensor, x_regime: torch.Tensor) -> torch.Tensor:
        informed_embedding = self.informed_trader_module(x_informed)
        noise_embedding = self.noise_trader_module(x_noise)
        market_maker_embedding = self.market_maker_module(x_mm)
        regime_embedding = self.regime_detection_module(x_regime)
        participant_embeddings = torch.stack([informed_embedding, noise_embedding, market_maker_embedding], dim=1)
        query = regime_embedding.unsqueeze(1)
        context_vector, _ = self.attention(query=query, key=participant_embeddings, value=participant_embeddings, need_weights=False)
        context_vector = context_vector.squeeze(1)
        return self.output_layer(context_vector)

# =============================================================================
# All test code from the project, consolidated into one file.
# =============================================================================

class TestFinalProject(unittest.TestCase):

    def test_data_pipeline(self):
        print("\n--- Testing Data Pipeline ---")
        reader = MarketDataReader(assets=['BTC-USD'], start_date='2023-01-01', end_date='2023-12-31')
        final_data = reader.load_and_process()
        self.assertIn('BTC-USD', final_data)
        btc_data = final_data['BTC-USD']
        self.assertIsInstance(btc_data, pd.DataFrame)
        self.assertFalse(btc_data.empty)
        self.assertFalse(btc_data.isnull().any().any())
        print("✓ Data pipeline test passed.")

    def test_participant_modules(self):
        print("\n--- Testing Participant Modules ---")
        module = InformedTraderModule(feature_dim=5, hidden_dim=16)
        sample_input = torch.randn(4, 20, 5)
        output = module(sample_input)
        self.assertEqual(output.shape, (4, 16))
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(module.output_layer.weight.grad)
        print("✓ Participant modules are trainable and have correct shapes.")

    def test_full_model_smoke_test(self):
        print("\n--- Testing Full MarketMind Model ---")
        model = MarketMindModel(informed_trader_features=2, noise_trader_features=1, market_maker_features=6, regime_features=2)
        x_inf = torch.randn(4, 30, 2)
        x_noise = torch.randn(4, 30, 1)
        x_mm = torch.randn(4, 30, 6)
        x_reg = torch.randn(4, 30, 2)
        prediction = model(x_inf, x_noise, x_mm, x_reg)
        self.assertEqual(prediction.shape, (4, 1))
        loss = prediction.sum()
        loss.backward()
        self.assertIsNotNone(model.output_layer.weight.grad)
        print("✓ Full model is trainable and has correct shapes.")

if __name__ == '__main__':
    print("="*60)
    print("Executing Consolidated Project Test Suite")
    print("="*60)
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFinalProject))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    # The script will exit here in a real run, but this allows programmatic check
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
