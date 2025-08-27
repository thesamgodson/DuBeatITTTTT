import pandas as pd
import numpy as np

def inject_flash_crash(df: pd.DataFrame, date: str, price_drop: float = 0.20, volume_spike: float = 5.0) -> pd.DataFrame:
    """
    Injects a synthetic flash crash event into a time series DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Close' and 'Volume' columns.
        date (str): The date to inject the crash, in 'YYYY-MM-DD' format.
        price_drop (float): The percentage drop in price (e.g., 0.20 for 20%).
        volume_spike (float): The factor to multiply the volume by (e.g., 5.0 for 5x).

    Returns:
        pd.DataFrame: A new DataFrame with the injected shock.
    """
    shock_df = df.copy()
    target_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

    # Create a boolean mask for the target date
    date_mask = shock_df.index.strftime('%Y-%m-%d') == target_date_str

    if not date_mask.any():
        print(f"Shock date {date} not found in DataFrame index. No shock injected.")
        return df

    # Apply the shock using the mask
    shock_df.loc[date_mask, 'Close'] *= (1 - price_drop)
    shock_df.loc[date_mask, 'Volume'] *= volume_spike

    print(f"Injected a {price_drop*100}% price drop and {volume_spike}x volume spike on {target_date_str}.")

    return shock_df

def inject_volume_spike(df: pd.DataFrame, date: str, volume_spike: float = 5.0) -> pd.DataFrame:
    """Injects a synthetic volume spike without a price change."""
    shock_df = df.copy()
    target_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    date_mask = shock_df.index.strftime('%Y-%m-%d') == target_date_str
    if not date_mask.any():
        print(f"Shock date {date} not found. No shock injected.")
        return df
    shock_df.loc[date_mask, 'Volume'] *= volume_spike
    print(f"Injected a {volume_spike}x volume spike on {target_date_str}.")
    return shock_df

def inject_trend_reversal(df: pd.DataFrame, date: str, duration: int = 5) -> pd.DataFrame:
    """Injects a synthetic trend reversal over a few days."""
    shock_df = df.copy()
    start_date = pd.to_datetime(date)

    # Get the slice of the dataframe to modify
    reversal_dates = pd.date_range(start=start_date, periods=duration, freq='D')

    # Get the price before the reversal starts
    price_before = shock_df.loc[shock_df.index < start_date, 'Close'].iloc[-1]

    # Create a linear reversal
    reversal_prices = np.linspace(price_before, price_before * (1 - 0.10), duration) # 10% drop

    for i, d in enumerate(reversal_dates):
        date_str = d.strftime('%Y-%m-%d')
        if date_str in shock_df.index.strftime('%Y-%m-%d'):
            shock_df.loc[date_str, 'Close'] = reversal_prices[i]

    print(f"Injected a {duration}-day trend reversal starting on {date}.")
    return shock_df

if __name__ == '__main__':
    # Example usage requires a sample dataframe
    # This will be tested via the profiling script
    pass
