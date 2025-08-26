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

    # --- 2. Model Initialization ---
    use_som = model_config.get('use_som_gating', False)
    feature_extractor = None

    if use_som:
        print("Training SOM...")
        close_price_idx = scaled_feature_cols.index('Close_scaled')
        som_training_data = data_loader.X[:, :, close_price_idx].numpy()
        som = SimpleSOM(input_len=model_config['sequence_length'], grid_size=(10, 10))
        som.train(som_training_data, num_iteration=1000)
        feature_extractor = SOMFeatureExtractor(som)

    print("Initializing NBEATS model...")
    model = NBEATSWithSOM(
        som_feature_extractor=feature_extractor,
        use_som_gating=use_som,
        sequence_length=model_config['sequence_length'],
        forecast_horizon=model_config['forecast_horizon'],
        n_stacks=2,
        n_blocks=3,
        n_layers=4,
        layer_width=256,
        num_features=len(feature_cols)
    )

    # --- 3. Model Training ---
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

    # --- 4. Evaluation ---
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

    y_pred_prices = target_scaler.inverse_transform(y_pred_scaled)

    test_set_size = len(y_true_scaled)
    test_start_index = len(btc_df) - test_set_size

    # Get the actual returns for the test period
    actual_returns = btc_df['Close'].pct_change().iloc[test_start_index:].values

    # To calculate predicted returns, we need the price from the previous day for each prediction
    prev_prices = btc_df['Close'].iloc[test_start_index-1:-1].values

    # We use the first step of the forecast horizon for our prediction
    predicted_returns = (y_pred_prices[:, 0] / prev_prices) - 1

    # Ensure lengths match for safety, though they should be correct
    min_len = min(len(predicted_returns), len(actual_returns))

    metrics = evaluator.calculate_all_metrics(
        predictions=predicted_returns[:min_len],
        actual_returns=actual_returns[:min_len]
    )

    print(f"Evaluation complete for {model_config['name']}.")
    return metrics

def main():
    model_configs = [
        {
            "name": "Pure N-BEATS (Close Only)",
            "feature_cols": ['Close'],
            "use_som_gating": False,
            "sequence_length": 30,
            "forecast_horizon": 5,
            "epochs": 30
        },
        {
            "name": "NBEATS-SOM (Close Only)",
            "feature_cols": ['Close'],
            "use_som_gating": True,
            "sequence_length": 30,
            "forecast_horizon": 5,
            "epochs": 30
        },
        {
            "name": "NBEATS-SOM (Close + Volume)",
            "feature_cols": ['Close', 'Volume'],
            "use_som_gating": True,
            "sequence_length": 30,
            "forecast_horizon": 5,
            "epochs": 30
        }
    ]

    print("Loading and preparing base data...")
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    if btc_df.empty:
        print("Failed to load data. Exiting.")
        return

    evaluator = FinancialMetricsEvaluator()

    results = {}
    for config in model_configs:
        metrics = run_model_evaluation(config, btc_df.copy(), evaluator)
        results[config['name']] = metrics

    print("\n\n" + "="*60)
    print("           MASTER MODEL EVALUATION REPORT")
    print("="*60)

    for model_name, metrics in results.items():
        print(f"\n--- Results for: {model_name} ---")
        evaluator.print_evaluation_report(metrics)

if __name__ == "__main__":
    main()
