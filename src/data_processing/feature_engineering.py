import pandas as pd
import numpy as np

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates return-based features."""
    df['Returns'] = df['Close'].pct_change()
    df['LogReturns'] = np.log(df['Close']).diff()
    return df

def add_volatility_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates volatility-based features."""
    df['Volatility'] = df['LogReturns'].rolling(window=window).std()

    # Average True Range (ATR) as another measure of volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=window).mean()

    return df

def add_microstructure_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculates features that serve as proxies for market microstructure concepts.
    """
    # Liquidity Proxy: High-Low spread relative to price. Higher means less liquid.
    df['LiquidityProxy'] = (df['High'] - df['Low']) / df['Close']

    # Informed Trading Proxy: Volume-weighted absolute returns.
    # High values suggest informed trades are moving the market.
    df['InformedTradingProxy'] = df['LogReturns'].abs() * df['Volume']

    # Noise Trading Proxy: A simple proxy could be days with high volume but low price movement.
    # We can calculate the rolling correlation between volume and absolute returns.
    # A lower correlation suggests more volume is not contributing to price discovery (i.e., noise).
    df['NoiseTradingProxy'] = df['Volume'].rolling(window=window).corr(df['LogReturns'].abs())

    return df

def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to the raw market data.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].

    Returns:
        pd.DataFrame: The DataFrame with all engineered features.
    """
    df = df.copy()
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_microstructure_features(df)

    # Drop rows with NaN values created during feature engineering
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
