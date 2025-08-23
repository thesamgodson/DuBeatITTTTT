import pandas as pd
import numpy as np

def create_basic_features(df):
    """
    Adds basic financial features to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'Close' price column.

    Returns:
        pd.DataFrame: DataFrame with added features: 'returns', 'log_returns',
                      'SMA_10', and 'SMA_30'.
    """
    if 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    # Calculate daily returns
    df['returns'] = df['Close'].pct_change()

    # Calculate log returns
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate Simple Moving Averages (SMAs)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()

    # Drop rows with NaN values created by feature engineering
    df.dropna(inplace=True)

    return df

def create_pattern_vectors(df, window_size=3):
    """
    Creates normalized price sequences for SOM input.

    This function generates overlapping windows of 'Close' prices and normalizes
    each window by dividing by its first element.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column.
        window_size (int): The size of the sliding window.

    Returns:
        np.ndarray: A numpy array of shape (n_samples, window_size) containing
                    the normalized patterns.
    """
    if 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    if len(df) < window_size:
        return np.array([])

    close_prices = df['Close'].values

    # Create overlapping windows
    shape = (close_prices.shape[0] - window_size + 1, window_size)
    strides = (close_prices.strides[0], close_prices.strides[0])
    pattern_windows = np.lib.stride_tricks.as_strided(close_prices, shape=shape, strides=strides)

    # Normalize each window by its first element to get the price sequence pattern
    # Adding a small epsilon to avoid division by zero, although price should be > 0
    normalized_patterns = pattern_windows / (pattern_windows[:, 0:1] + 1e-9)

    return normalized_patterns
