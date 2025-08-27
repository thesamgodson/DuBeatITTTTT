import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.data_processing.macro_loader import get_google_trends_data
from src.driftnet.model import DriftNet
from src.driftnet.causal import get_causal_features_lasso
from src.evaluation import FinancialMetricsEvaluator

def create_sequences(data, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon), 0])
    return np.array(X), np.array(y)

def main():
    # --- 1. Load and Prepare Data ---
    print("Loading all data sources...")
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    trends_df = get_google_trends_data(['bitcoin'], '2020-01-01', '2024-12-31')
    if btc_df.empty or trends_df.empty: return

    full_df = btc_df.join(trends_df, how='outer').ffill()
    full_df.sort_index(inplace=True)
    full_df.dropna(inplace=True)

    feature_cols = ['Close', 'Volume', 'bitcoin']
    target_col = 'Close'

    # --- 2. Rolling Window Evaluation ---
    start_date = full_df.index.min()
    end_date = full_df.index.max()
    train_window = pd.DateOffset(months=6)
    test_window = pd.DateOffset(months=2)
    current_date = start_date

    all_sharpe_ratios, expert_counts, causal_feature_log, fdd_alarm_log = [], [], [], []

    sequence_length, forecast_horizon = 30, 5
    initial_train_df = full_df.loc[start_date : start_date + train_window]
    X_initial, _ = create_sequences(initial_train_df[feature_cols].values, sequence_length, forecast_horizon)
    input_dim = X_initial.shape[1] * X_initial.shape[2]
    model = DriftNet(input_dim=input_dim, forecast_horizon=forecast_horizon)
    evaluator = FinancialMetricsEvaluator()
    consecutive_fdd_alarms, merge_counter = 0, 0

    while current_date + train_window + test_window <= end_date:
        train_start, train_end, test_end = current_date, current_date + train_window, current_date + train_window + test_window
        print(f"\n{'='*20} Running Window: {train_start.date()} to {test_end.date()} {'='*20}")

        train_df_unscaled = full_df.loc[train_start:train_end]

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        train_df_scaled = train_df_unscaled.copy()
        train_df_scaled[feature_cols] = feature_scaler.fit_transform(train_df_unscaled[feature_cols])
        target_scaler.fit(train_df_unscaled[[target_col]])

        X_train, y_train = create_sequences(train_df_scaled[feature_cols].values, sequence_length, forecast_horizon)

        if len(X_train) == 0:
            current_date += test_window
            continue

        selected_features = get_causal_features_lasso(X_train, y_train, feature_cols)
        causal_feature_log.append({"window_end": train_end.date(), "selected_features": selected_features})

        print("Training DriftNet...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(5):
            for i in range(len(X_train)):
                x_i = torch.from_numpy(X_train[i]).float().view(1, -1)
                y_i = torch.from_numpy(y_train[i]).float().view(1, -1)
                bmu_idx = model.get_bmu_and_update_som(x_i, epoch, 5)
                optimizer.zero_grad()
                y_pred = model.experts[bmu_idx](x_i)
                loss = criterion(y_pred, y_i)
                loss.backward()
                optimizer.step()

        print("Evaluating on test set...")
        model.eval()
        test_df_unscaled = full_df.loc[train_end:test_end]
        test_df_scaled_values = feature_scaler.transform(test_df_unscaled[feature_cols])
        X_test, y_test = create_sequences(test_df_scaled_values, sequence_length, forecast_horizon)

        predictions_scaled, forecast_errors = [], []
        with torch.no_grad():
            for i in range(len(X_test)):
                x_i = torch.from_numpy(X_test[i]).float().view(1, -1)
                y_i_true_scaled = torch.from_numpy(y_test[i]).float().view(1, -1)
                y_pred_scaled = model(x_i)
                predictions_scaled.append(y_pred_scaled.numpy())
                error = criterion(y_pred_scaled, y_i_true_scaled).item()
                forecast_errors.append(error)

        avg_window_error = np.mean(forecast_errors) if forecast_errors else 0
        if model.fdd.check_novelty(avg_window_error):
            fdd_alarm_log.append({"timestamp": test_end.date(), "avg_forecast_error": avg_window_error})
            consecutive_fdd_alarms += 1
        else:
            consecutive_fdd_alarms = 0

        if consecutive_fdd_alarms >= 3:
            print(f"--- FORECAST CRISIS! Spawning new expert. ---")
            last_x_numpy = X_test[-1].flatten()
            bmu = model.dynamic_som.find_bmu(last_x_numpy)
            model._spawn_new_expert(bmu, test_end.date(), last_x_numpy)
            consecutive_fdd_alarms = 0

        merge_counter += 1
        if merge_counter >= 6:
            model._merge_experts()
            merge_counter = 0

        # --- Financial Evaluation ---
        predictions_unscaled = target_scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, forecast_horizon))

        last_prices_unscaled = test_df_unscaled[target_col].values[sequence_length-1:-forecast_horizon]

        # We use the first step of the forecast for our signal
        signals = (predictions_unscaled[:, 0] > last_prices_unscaled).astype(int)

        # The actual return is the return of the day we are predicting
        actual_returns = test_df_unscaled[target_col].pct_change().values[sequence_length:]

        min_len = min(len(signals), len(actual_returns))
        metrics = evaluator.calculate_all_metrics(predictions=signals[:min_len], actual_returns=actual_returns[:min_len])
        all_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
        expert_counts.append(model.dynamic_som.num_nodes)

        current_date += test_window

    # --- Final Report and Saving Artifacts ---
    print("\n\n" + "="*60)
    print("           DRIFTNET v0.2 ROLLING EVALUATION REPORT")
    print("="*60)
    print(f"Average Sharpe Ratio across all windows: {np.mean(all_sharpe_ratios):.4f}")
    print(f"Standard Deviation of Sharpe Ratio: {np.std(all_sharpe_ratios):.4f}")
    print(f"Expert counts per window: {expert_counts}")
    if expert_counts:
        print(f"Final number of experts: {expert_counts[-1]}")

    # Save the logs
    os.makedirs('results', exist_ok=True)
    if model.birth_log:
        birth_log_df = pd.DataFrame(model.birth_log)
        birth_log_df.to_csv('results/expert_birth_log.csv', index=False)
        print("\nExpert birth log saved to 'results/expert_birth_log.csv'")
    if causal_feature_log:
        causal_df = pd.DataFrame(causal_feature_log)
        causal_df.to_csv('results/causal_feature_log.csv', index=False)
        print("Causal feature log saved to 'results/causal_feature_log.csv'")
    if fdd_alarm_log:
        fdd_df = pd.DataFrame(fdd_alarm_log)
        fdd_df.to_csv('results/fdd_alarm_log.csv', index=False)
        print("FDD alarm log saved to 'results/fdd_alarm_log.csv'")

if __name__ == "__main__":
    main()
