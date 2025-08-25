import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add the src directory to the Python path for sibling imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.feature_engineering import process_market_data

class MarketDataReader:
    """
    Handles loading raw market data from yfinance and applying feature engineering.
    """
    def __init__(self, assets: List[str], start_date: str, end_date: str):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date

    def load_and_process(self) -> Dict[str, pd.DataFrame]:
        """
        Loads data for all configured assets and processes them.

        Returns:
            A dictionary where keys are asset tickers and values are the
            processed pandas DataFrames with all features.
        """
        processed_data = {}
        for asset in self.assets:
            try:
                print(f"Loading and processing data for {asset}...")
                # 1. Load raw data
                raw_df = yf.download(asset, start=self.start_date, end=self.end_date, auto_adjust=True)

                # FIX: Flatten MultiIndex columns returned by yfinance
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)

                if raw_df.empty:
                    print(f"Warning: No data found for {asset}. Skipping.")
                    continue

                # 2. Apply feature engineering pipeline
                processed_df = process_market_data(raw_df)

                processed_data[asset] = processed_df
                print(f"✓ Successfully processed {asset}, {len(processed_df)} samples created.")

            except Exception as e:
                print(f"Error processing {asset}: {e}")

        return processed_data

class Aligner:
    """
    Creates supervised learning datasets from processed time series data.
    """
    def __init__(self, backcast_length: int, forecast_horizon: int):
        self.backcast_length = backcast_length
        self.forecast_horizon = forecast_horizon

    def create_sequences(self, df: pd.DataFrame, target_column: str = 'LogReturns') -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences of features (X) and targets (y).

        Args:
            df (pd.DataFrame): The processed DataFrame with all features.
            target_column (str): The name of the column to be used as the prediction target.

        Returns:
            A tuple containing:
            - X: A numpy array of shape [num_samples, backcast_length, num_features].
            - y: A numpy array of shape [num_samples, forecast_horizon].
        """
        X, y = [], []

        # The target is a single column
        target_array = df[target_column].values

        # The features are all other columns (except Date)
        feature_columns = [col for col in df.columns if col not in ['Date', target_column]]
        feature_array = df[feature_columns].values

        for i in range(self.backcast_length, len(df) - self.forecast_horizon + 1):
            # The input sequence is the window of features ending at the previous time step
            X.append(feature_array[i - self.backcast_length : i])

            # The target is the value at the current time step `i` (predicting one step ahead)
            y.append(target_array[i : i + self.forecast_horizon])

        return np.array(X), np.array(y)
