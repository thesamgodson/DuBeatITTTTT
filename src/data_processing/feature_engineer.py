import pandas as pd
import numpy as np

def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic financial features to the DataFrame.
    """
    if 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()

    df.dropna(inplace=True)
    return df

def create_pattern_vectors(data, window_size=3) -> np.ndarray:
    """
    Creates normalized price sequences for SOM input from a DataFrame or numpy array.
    """
    if isinstance(data, pd.DataFrame):
        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")
        close_prices = data['Close'].values
    elif isinstance(data, np.ndarray):
        close_prices = data
    else:
        raise TypeError("Input data must be a pandas DataFrame or a numpy array.")

    if len(close_prices) < window_size:
        return np.array([])

    shape = (close_prices.shape[0] - window_size + 1, window_size)
    strides = (close_prices.strides[0], close_prices.strides[0])
    pattern_windows = np.lib.stride_tricks.as_strided(close_prices, shape=shape, strides=strides)

    # Normalize each window by its first element
    normalized_patterns = pattern_windows / (pattern_windows[:, 0:1] + 1e-9)

    return normalized_patterns
