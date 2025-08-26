import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.data_processing.data_processor import TimeSeriesDataLoader
from src.som_analysis.som_trainer import SimpleSOM
from src.som_analysis.feature_extractor import SOMFeatureExtractor
from src.nbeats.nbeats_model import NBEATSWithSOM
from src.evaluation import FinancialMetricsEvaluator

def run_model_evaluation(model_config: dict, btc_df: pd.DataFrame, evaluator: FinancialMetricsEvaluator) -> dict:
    """
    Trains a model based on the given configuration and returns its financial metrics.
    """
    print(f"\n{'='*20} Running Evaluation for: {model_config['name']} {'='*20}")

    # --- 1. Data Preparation ---
    feature_cols = model_config['feature_cols']
    target_col = 'Close'

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_size = int(0.8 * len(btc_df))

    feature_scaler.fit(btc_df[feature_cols][:train_size])
    target_scaler.fit(btc_df[[target_col]][:train_size])

    scaled_features = feature_scaler.transform(btc_df[feature_cols])

    scaled_df = btc_df.copy()
    scaled_feature_cols = [f"{col}_scaled" for col in feature_cols]
    for i, col_name in enumerate(scaled_feature_cols):
        scaled_df[col_name] = scaled_features[:, i]

    data_loader = TimeSeriesDataLoader(
        df=scaled_df,
        sequence_length=model_config['sequence_length'],
        forecast_horizon=model_config['forecast_horizon'],
        train_split=0.8,
        batch_size=64,
        feature_cols=scaled_feature_cols,
        target_col=f"{target_col}_scaled"
    )
    train_loader, test_loader = data_loader.get_loaders()

    # --- 2. SOM Training ---
    print("Training SOM...")
    close_price_idx = scaled_feature_cols.index('Close_scaled')
    som_training_data = data_loader.X[:, :, close_price_idx].numpy()
    som = SimpleSOM(input_len=model_config['sequence_length'], grid_size=(10, 10))
    som.train(som_training_data, num_iteration=1000)

    # --- 3. Model Initialization ---
    print("Initializing NBEATSWithSOM model...")
    feature_extractor = SOMFeatureExtractor(som)
    model = NBEATSWithSOM(
        som_feature_extractor=feature_extractor,
        sequence_length=model_config['sequence_length'],
        forecast_horizon=model_config['forecast_horizon'],
        n_stacks=2,
        n_blocks=3,
        n_layers=4,
        layer_width=256,
        num_features=len(feature_cols)
    )

    # --- 4. Model Training ---
    print("Training NBEATS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(model_config['epochs']):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{model_config['epochs']}, Loss: {loss.item():.4f}")

    # --- 5. Evaluation ---
    print("Evaluating model...")
    model.eval()
    all_y_pred_scaled = []
    all_y_true_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            all_y_pred_scaled.append(y_pred.cpu().numpy())
            all_y_true_scaled.append(y_batch.cpu().numpy())

    y_pred_scaled = np.concatenate(all_y_pred_scaled)
    y_true_scaled = np.concatenate(all_y_true_scaled)

    # Inverse transform to get actual price predictions
    y_pred_prices = target_scaler.inverse_transform(y_pred_scaled)
    y_true_prices = target_scaler.inverse_transform(y_true_scaled)

    # The evaluator needs returns, not prices.
    # We need the actual returns of the test set.
    test_set_size = len(y_true_prices)
    actual_returns = btc_df[target_col].pct_change().dropna().values[-test_set_size:]

    # For predictions, we need to convert price predictions to return predictions.
    # This is tricky because we predict a sequence. Let's use the first step forecast.
    predicted_returns = (y_pred_prices[:, 0] / btc_df[target_col].values[-test_set_size-1:-1]) - 1

    # Ensure lengths match
    min_len = min(len(predicted_returns), len(actual_returns))

    metrics = evaluator.calculate_all_metrics(
        predictions=predicted_returns[:min_len],
        actual_returns=actual_returns[:min_len]
    )

    print(f"Evaluation complete for {model_config['name']}.")
    return metrics

def main():
    # Define model configurations to test
    model_configs = [
        {
            "name": "NBEATS-SOM (Close Only)",
            "feature_cols": ['Close'],
            "sequence_length": 30,
            "forecast_horizon": 5,
            "epochs": 30
        },
        {
            "name": "NBEATS-SOM (Close + Volume)",
            "feature_cols": ['Close', 'Volume'],
            "sequence_length": 30,
            "forecast_horizon": 5,
            "epochs": 30
        }
    ]

    # Load data once
    print("Loading and preparing base data...")
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    if btc_df.empty:
        print("Failed to load data. Exiting.")
        return

    # Initialize evaluator
    evaluator = FinancialMetricsEvaluator()

    results = {}
    for config in model_configs:
        metrics = run_model_evaluation(config, btc_df.copy(), evaluator)
        results[config['name']] = metrics

    # --- Print Final Report ---
    print("\n\n" + "="*60)
    print("           MASTER MODEL EVALUATION REPORT")
    print("="*60)

    for model_name, metrics in results.items():
        print(f"\n--- Results for: {model_name} ---")
        evaluator.print_evaluation_report(metrics)

    # Here you could also format the results for the README

if __name__ == "__main__":
    main()
