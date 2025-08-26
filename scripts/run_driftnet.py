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
    feature_importance_log = []

    # We initialize the model once, and it evolves over the windows.
    # This is a more realistic simulation of a live model.
    initial_train_df = full_df.loc[start_date : start_date + train_window]
    X_initial, _ = create_sequences(initial_train_df[feature_cols].values, 30, 5)
    input_dim = X_initial.shape[1] * X_initial.shape[2]
    model = DriftNet(input_dim=input_dim, forecast_horizon=5, novelty_threshold=1.1)

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

        # --- Train Model on Current Window ---
        print(f"Training DriftNet on data from {train_start.date()} to {train_end.date()}...")

        # Pre-train novelty detector on the new training data
        train_tensor_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float())
        train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=32)
        model.train_novelty_detector(train_loader, epochs=5)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(10):
            for i in range(len(X_train)):
                x_i = torch.from_numpy(X_train[i]).float().view(1, -1)
                y_i = torch.from_numpy(y_train[i]).float().view(1, -1)
                bmu_idx = model.get_bmu_and_update_som(x_i, epoch, 10)
                optimizer.zero_grad()
                y_pred = model.experts[bmu_idx](x_i)
                loss = criterion(y_pred, y_i)
                loss.backward()
                optimizer.step()

        # --- Evaluation on Test Set (with novelty detection) ---
        print(f"Evaluating on test set from {train_end.date()} to {test_end.date()}...")
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(X_test)):
                x_i = torch.from_numpy(X_test[i]).float().view(1, -1)
                current_sample_date = test_df.index[i + 30 - 1]
                y_pred = model(x_i, current_sample_date)
                predictions.append(y_pred.numpy())

        mse = np.mean((np.array(predictions).flatten() - y_test.flatten())**2)
        print(f"Test MSE for this window: {mse:.4f}")
        all_results.append(mse)
        expert_counts.append(model.dynamic_som.num_nodes)

        # --- Log Feature Importance for this window ---
        num_features = len(feature_cols)
        seq_len = 30
        for expert_id, expert in enumerate(model.experts):
            weights = expert.weight.detach().cpu().numpy()
            weights_reshaped = weights.reshape(-1, seq_len, num_features)
            for i, feature_name in enumerate(feature_cols):
                feature_importance = np.mean(np.abs(weights_reshaped[:, :, i]))
                feature_importance_log.append({
                    "window_end_date": train_end.date(),
                    "expert_id": expert_id,
                    "feature": feature_name,
                    "importance": feature_importance
                })

        current_date += test_window

    # --- Final Report and Saving Artifacts ---
    print("\n\n" + "="*60)
    print("           DRIFTNET ROLLING EVALUATION REPORT")
    print("="*60)
    print(f"Average Test MSE across all windows: {np.mean(all_results):.4f}")
    print(f"Expert counts per window: {expert_counts}")
    if expert_counts:
        print(f"Final number of experts: {expert_counts[-1]}")

    # Save the logs
    os.makedirs('results', exist_ok=True)
    if model.birth_log:
        birth_log_df = pd.DataFrame(model.birth_log)
        birth_log_df.to_csv('results/expert_birth_log.csv', index=False)
        print("\nExpert birth log saved to 'results/expert_birth_log.csv'")
    if feature_importance_log:
        importance_df = pd.DataFrame(feature_importance_log)
        importance_df.to_csv('results/expert_feature_importance.csv', index=False)
        print("Feature importance log saved to 'results/expert_feature_importance.csv'")

if __name__ == "__main__":
    main()
