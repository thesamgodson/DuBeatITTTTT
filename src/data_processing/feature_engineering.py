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
    df['LiquidityProxy'] = (df['High'] - df['Low']) / df['Close']
    df['InformedTradingProxy'] = df['LogReturns'].abs() * df['Volume']
    # The rolling correlation can produce NaNs if the window has zero variance. Fill with 0.
    df['NoiseTradingProxy'] = df['Volume'].rolling(window=window).corr(df['LogReturns'].abs()).fillna(0)
    return df

def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to the raw market data.
    """
    df = df.copy()
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_microstructure_features(df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
