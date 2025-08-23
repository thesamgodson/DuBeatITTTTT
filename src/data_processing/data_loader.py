import pandas as pd
import yfinance as yf

def load_bitcoin_data(start_date='2014-01-01', end_date='2024-12-31'):
    """
    Downloads and performs basic cleaning of Bitcoin data from yfinance.

    Args:
        start_date (str): The start date for the data download (YYYY-MM-DD).
        end_date (str): The end date for the data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
                      Returns an empty DataFrame if data download fails.
    """
    try:
        # Download Bitcoin data from Yahoo Finance
        df = yf.download('BTC-USD', start=start_date, end=end_date)

        if df.empty:
            print(f"Warning: No data downloaded for the specified date range {start_date} to {end_date}.")
            return pd.DataFrame()

        # yfinance may return a MultiIndex column. Flatten it by dropping the 'Ticker' level.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df.reset_index(inplace=True)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Check if all required columns are present after potential cleaning
        for col in required_columns:
            if col not in df.columns:
                available_cols = df.columns.tolist()
                raise ValueError(f"Missing expected column '{col}'. Available columns: {available_cols}")

        # Select only the required columns
        df = df[required_columns]

        # Handle potential missing values in the 'Close' column
        df['Close'] = df['Close'].ffill().bfill()

        # Drop any rows that still have NaNs in any of the required columns
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"An error occurred while loading Bitcoin data: {e}")
        return pd.DataFrame()
