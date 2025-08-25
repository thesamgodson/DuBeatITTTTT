import pandas as pd
import yfinance as yf

def load_bitcoin_data(start_date='2014-01-01', end_date='2024-12-31'):
    """
    Downloads and performs basic cleaning of Bitcoin data from yfinance.
    """
    try:
        df = yf.download('BTC-USD', start=start_date, end=end_date)

        if df.empty:
            print(f"Warning: No data downloaded for the specified date range {start_date} to {end_date}.")
            return pd.DataFrame()

        # Handle MultiIndex columns returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df.reset_index(inplace=True)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing expected column '{col}'. Available columns: {df.columns.tolist()}")

        df = df[required_columns]

        # Handle potential missing values
        df['Close'] = df['Close'].ffill().bfill()
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"An error occurred while loading Bitcoin data: {e}")
        return pd.DataFrame()
