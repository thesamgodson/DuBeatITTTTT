import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.data_processing.macro_loader import get_google_trends_data
from src.data_processing.synthetic_shocks import inject_flash_crash, inject_volume_spike, inject_trend_reversal
from src.driftnet.autoencoder import AutoEncoder

def create_sequences(data, sequence_length):
    """Creates sequences from the input data."""
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:(i + sequence_length)])
    return np.array(X)

def run_profiling(full_df: pd.DataFrame, test_case: str):
    """Runs the rolling window profiling for a given dataframe and test case."""
    print(f"\n--- Running Profiling for Test Case: {test_case} ---")

    start_date = full_df.index.min()
    end_date = full_df.index.max()
    train_window = pd.DateOffset(months=6)
    test_window = pd.DateOffset(months=2)
    current_date = start_date

    all_errors_with_ts = []
    sequence_length = 30

    while current_date + train_window + test_window <= end_date:
        train_start = current_date
        train_end = current_date + train_window
        test_end = train_end + test_window

        train_df = full_df.loc[train_start:train_end]
        test_df = full_df.loc[train_end:test_end]

        X_train = create_sequences(train_df[['Close', 'Volume', 'bitcoin']].values, sequence_length)
        X_test = create_sequences(test_df[['Close', 'Volume', 'bitcoin']].values, sequence_length)

        if len(X_train) == 0 or len(X_test) == 0:
            current_date += test_window
            continue

        input_dim = X_train.shape[1] * X_train.shape[2]
        autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=10)
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_tensor = torch.from_numpy(X_train).float().view(X_train.shape[0], -1)
        train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=32)

        autoencoder.train()
        for epoch in range(5):
            for data_batch in train_loader:
                optimizer.zero_grad()
                reconstructed = autoencoder(data_batch)
                loss = criterion(reconstructed, data_batch)
                loss.backward()
                optimizer.step()

        test_tensor = torch.from_numpy(X_test).float().view(X_test.shape[0], -1)
        errors = autoencoder.get_reconstruction_error(test_tensor).numpy()

        test_end_dates = test_df.index[sequence_length-1:]
        min_len = min(len(errors), len(test_end_dates))
        for i in range(min_len):
            all_errors_with_ts.append({'timestamp': test_end_dates[i], 'error': errors[i]})

        current_date += test_window

    return pd.DataFrame(all_errors_with_ts).set_index('timestamp')

def plot_distribution(error_df, save_path):
    """Plots the histogram of reconstruction errors."""
    print("\n--- Plotting Baseline Error Distribution ---")
    errors_array = error_df['error'].values
    mean_error = np.mean(errors_array)
    std_error = np.std(errors_array)
    threshold_2std = mean_error + 2 * std_error

    plt.figure(figsize=(12, 6))
    plt.hist(errors_array, bins=100, alpha=0.75)
    plt.axvline(threshold_2std, color='r', linestyle='--', label=f'Mean + 2*std ({threshold_2std:.4f})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Baseline Reconstruction Errors')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Distribution plot saved to {save_path}")
    return threshold_2std

def plot_shock_timeseries(error_df, shock_date, test_case, save_path):
    """Plots the time series of reconstruction errors for a shock test."""
    print(f"\n--- Plotting Error Timeseries for {test_case} ---")
    plt.figure(figsize=(15, 7))
    plt.plot(error_df.index, error_df['error'], label='Reconstruction Error')
    plt.axvline(pd.to_datetime(shock_date, utc=True), color='r', linestyle='--', label=f'Synthetic Shock')
    plt.xlabel('Date')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title(f'Novelty Detector Response to {test_case}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(save_path)
    plt.close()
    print(f"Time series plot saved to {save_path}")

def main():
    os.makedirs('results', exist_ok=True)

    # --- Load Base Data ---
    print("Loading base data...")
    from sklearn.preprocessing import MinMaxScaler
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    trends_df = get_google_trends_data(['bitcoin'], '2020-01-01', '2024-12-31')

    if btc_df.empty or trends_df.empty:
        print("Failed to load necessary data. Exiting calibration.")
        return

    # --- Test Case 1: Baseline (No Shock) ---
    full_df_base = btc_df.join(trends_df, how='outer').ffill()
    full_df_base.dropna(inplace=True)
    scaler_base = MinMaxScaler()
    full_df_base[['Close', 'Volume', 'bitcoin']] = scaler_base.fit_transform(full_df_base[['Close', 'Volume', 'bitcoin']])
    baseline_errors_df = run_profiling(full_df_base, "Baseline")
    calibrated_threshold = plot_distribution(baseline_errors_df, 'results/reconstruction_error_distribution.png')
    print(f"\n>>> Calibrated Novelty Threshold (Mean + 2*std): {calibrated_threshold:.6f} <<<")

    # --- Test Case 2: Flash Crash ---
    shock_date_crash = "2022-05-12"
    df_crash = inject_flash_crash(btc_df, date=shock_date_crash)
    full_df_crash = df_crash.join(trends_df, how='outer').ffill()
    full_df_crash.dropna(inplace=True)
    scaler_crash = MinMaxScaler()
    full_df_crash[['Close', 'Volume', 'bitcoin']] = scaler_crash.fit_transform(full_df_crash[['Close', 'Volume', 'bitcoin']])
    crash_errors_df = run_profiling(full_df_crash, "Flash Crash")
    plot_shock_timeseries(crash_errors_df, shock_date_crash, "Flash Crash", 'results/shock_test_flash_crash.png')

    # --- Test Case 3: Volume Spike ---
    shock_date_vol = "2021-01-20"
    df_vol = inject_volume_spike(btc_df, date=shock_date_vol)
    full_df_vol = df_vol.join(trends_df, how='outer').ffill()
    full_df_vol.dropna(inplace=True)
    scaler_vol = MinMaxScaler()
    full_df_vol[['Close', 'Volume', 'bitcoin']] = scaler_vol.fit_transform(full_df_vol[['Close', 'Volume', 'bitcoin']])
    vol_errors_df = run_profiling(full_df_vol, "Volume Spike")
    plot_shock_timeseries(vol_errors_df, shock_date_vol, "Volume Spike", 'results/shock_test_volume_spike.png')

if __name__ == "__main__":
    main()
