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
from src.data_processing.synthetic_shocks import inject_flash_crash
from src.driftnet.autoencoder import AutoEncoder

def create_sequences(data, sequence_length):
    """Creates sequences from the input data."""
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:(i + sequence_length)])
    return np.array(X)

def main():
    # --- 1. Load and Prepare Data ---
    print("Loading all data sources...")
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    trends_df = get_google_trends_data(['bitcoin'], '2020-01-01', '2024-12-31')

    if btc_df.empty or trends_df.empty:
        print("Failed to load initial data. Exiting.")
        return

    # --- Inject Synthetic Shock ---
    shock_date = "2022-05-12"
    btc_df = inject_flash_crash(btc_df, date=shock_date, price_drop=0.25, volume_spike=6.0)

    # Ensure index is a clean DatetimeIndex before joining
    btc_df.index = pd.to_datetime(btc_df.index, utc=True)
    trends_df.index = pd.to_datetime(trends_df.index, utc=True)

    full_df = btc_df.join(trends_df, how='outer').ffill()
    full_df.sort_index(inplace=True)
    full_df.dropna(inplace=True)

    feature_cols = ['Close', 'Volume', 'bitcoin']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])

    print("Data loaded and prepared.")

    # --- 2. Rolling Window Error Profiling ---
    start_date = full_df.index.min()
    end_date = full_df.index.max()
    train_window = pd.DateOffset(months=6)
    test_window = pd.DateOffset(months=2)
    current_date = start_date

    error_timeseries = []
    sequence_length = 30

    while current_date + train_window + test_window <= end_date:
        train_start = current_date
        train_end = current_date + train_window
        test_end = train_end + test_window

        train_df = full_df.loc[train_start:train_end]
        test_df = full_df.loc[train_end:test_end]

        X_train = create_sequences(train_df[feature_cols].values, sequence_length)
        X_test = create_sequences(test_df[feature_cols].values, sequence_length)

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

        # Get timestamps for the test errors
        test_end_dates = test_df.index[sequence_length-1:]

        # Ensure lengths match
        min_len = min(len(errors), len(test_end_dates))
        for i in range(min_len):
            error_timeseries.append({'timestamp': test_end_dates[i], 'error': errors[i]})

        current_date += test_window

    # --- 3. Plot Error Timeseries ---
    print("\n--- Plotting Error Timeseries ---")
    error_df = pd.DataFrame(error_timeseries).set_index('timestamp')

    plt.figure(figsize=(15, 7))
    plt.plot(error_df.index, error_df['error'], label='Reconstruction Error', alpha=0.8)
    plt.axvline(pd.to_datetime(shock_date, utc=True), color='r', linestyle='--', label=f'Synthetic Shock ({shock_date})')

    plt.xlabel('Date')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Novelty Detector Error Time Series with Synthetic Shock')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Use log scale to see spikes better

    os.makedirs('results', exist_ok=True)
    save_path = 'results/shock_test_error_timeseries.png'
    plt.savefig(save_path)
    print(f"\nTime series plot saved to {save_path}")

if __name__ == "__main__":
    main()
