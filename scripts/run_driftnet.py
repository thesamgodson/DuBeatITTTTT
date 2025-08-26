import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.data_processing.macro_loader import get_google_trends_data
from src.driftnet.model import DriftNet
# from src.evaluation import FinancialMetricsEvaluator # Will be used later

def create_sequences(data, sequence_length, forecast_horizon):
    """Creates sequences from the input data."""
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon), 0]) # Target is 'Close'
    return np.array(X), np.array(y)

def main():
    # --- 1. Load and Prepare Data ---
    print("Loading all data sources...")
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    trends_df = get_google_trends_data(['bitcoin'], '2020-01-01', '2024-12-31')

    if btc_df.empty or trends_df.empty:
        print("Failed to load initial data. Exiting.")
        return

    # Ensure index is a clean DatetimeIndex before joining
    btc_df.index = pd.to_datetime(btc_df.index, utc=True)
    trends_df.index = pd.to_datetime(trends_df.index, utc=True)

    # Merge dataframes
    full_df = btc_df.join(trends_df, how='outer').ffill()
    full_df.sort_index(inplace=True)
    full_df.dropna(inplace=True)

    feature_cols = ['Close', 'Volume', 'bitcoin']

    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])

    print("Data loaded and prepared.")

    # --- 2. Rolling Window Evaluation ---
    start_date = full_df.index.min()
    end_date = full_df.index.max()

    train_window = pd.DateOffset(months=6)
    test_window = pd.DateOffset(months=2) # Increased to 2 months

    current_date = start_date

    all_results = []
    expert_counts = []

    while current_date + train_window + test_window <= end_date:
        train_start = current_date
        train_end = current_date + train_window
        test_end = train_end + test_window

        print(f"\n{'='*20} Running Window: {train_start.date()} to {test_end.date()} {'='*20}")

        # --- Get data for the current window ---
        train_df = full_df.loc[train_start:train_end]
        test_df = full_df.loc[train_end:test_end]

        X_train, y_train = create_sequences(train_df[feature_cols].values, 30, 5)
        X_test, y_test = create_sequences(test_df[feature_cols].values, 30, 5)

        if len(X_train) == 0 or len(X_test) == 0:
            print("Not enough data in this window. Skipping.")
            current_date += test_window
            continue

        # --- Initialize and Train Model ---
        print("Initializing and training DriftNet for this window...")
        input_dim = X_train.shape[1] * X_train.shape[2] # seq_len * num_features

        model = DriftNet(input_dim=input_dim, forecast_horizon=5)

        # Pre-train novelty detector
        train_tensor_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float())
        train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=32)
        model.train_novelty_detector(train_loader, epochs=5)

        # Main training loop
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(10): # Simplified training for now
            for i in range(len(X_train)):
                x_i = torch.from_numpy(X_train[i]).float().view(1, -1)
                y_i = torch.from_numpy(y_train[i]).float().view(1, -1)

                # Update SOM and get BMU for training the experts
                bmu_idx = model.get_bmu_and_update_som(x_i, epoch, 10)

                # Train the expert associated with the BMU
                optimizer.zero_grad()
                # We need to get the prediction from the specific expert to train it
                y_pred = model.experts[bmu_idx](x_i)
                loss = criterion(y_pred, y_i)
                loss.backward()
                optimizer.step()

        # --- Evaluation on Test Set (with novelty detection) ---
        print("Evaluating on test set...")
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(X_test)):
                x_i = torch.from_numpy(X_test[i]).float().view(1, -1)
                # The forward pass now handles novelty detection and expert spawning
                y_pred = model(x_i)
                predictions.append(y_pred.numpy())

        # Store results (simplified for now)
        mse = np.mean((np.array(predictions).flatten() - y_test.flatten())**2)
        print(f"Test MSE for this window: {mse:.4f}")
        all_results.append(mse)
        expert_counts.append(model.dynamic_som.num_nodes)

        # Move to the next window
        current_date += test_window

    print("\n\n" + "="*60)
    print("           DRIFTNET ROLLING EVALUATION REPORT")
    print("="*60)
    print(f"Average Test MSE across all windows: {np.mean(all_results):.4f}")
    print(f"Expert counts per window: {expert_counts}")
    print(f"Final number of experts: {expert_counts[-1]}")

if __name__ == "__main__":
    main()
