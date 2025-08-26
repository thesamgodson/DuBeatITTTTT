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

def calculate_directional_accuracy(y_true, y_pred):
    true_diff = np.diff(y_true, axis=1)
    pred_diff = np.diff(y_pred, axis=1)
    correct_direction = (np.sign(true_diff) == np.sign(pred_diff))
    return np.mean(correct_direction) * 100 if correct_direction.size > 0 else 0

def main():
    # 1. Load and Prepare Data
    print("Loading data...")
    btc_df = load_bitcoin_data(start_date='2020-01-01', end_date='2024-12-31')
    if btc_df.empty:
        print("Failed to load data. Exiting.")
        return

    # --- Data Scaling ---
    print("Scaling data...")
    feature_cols = ['Close', 'Volume']
    target_col = 'Close'

    # Scaler for input features
    feature_scaler = StandardScaler()

    # Scaler for the target variable
    target_scaler = StandardScaler()

    train_size = int(0.8 * len(btc_df))

    # Fit scalers on the training data portion
    feature_scaler.fit(btc_df[feature_cols][:train_size])
    target_scaler.fit(btc_df[[target_col]][:train_size])

    # Transform the data
    scaled_features = feature_scaler.transform(btc_df[feature_cols])

    # Create new columns for scaled data
    scaled_feature_cols = [f"{col}_scaled" for col in feature_cols]
    for i, col_name in enumerate(scaled_feature_cols):
        btc_df[col_name] = scaled_features[:, i]

    # Use the scaled data for the model
    data_loader = TimeSeriesDataLoader(
        df=btc_df,
        sequence_length=30,
        forecast_horizon=5,
        train_split=0.8,
        batch_size=64,
        feature_cols=scaled_feature_cols,
        target_col='Close_scaled' # The target is the scaled close price
    )
    train_loader, test_loader = data_loader.get_loaders()

    # 2. Train SOM on the 'Close' price sequences
    print("Training SOM...")
    # We need the un-sequenced close prices for SOM training patterns
    # The shape of X is (num_samples, sequence_length, num_features)
    close_price_sequences = data_loader.X[:, :, scaled_feature_cols.index('Close_scaled')].numpy()
    som = SimpleSOM(input_len=30, grid_size=(10, 10), learning_rate=0.5, sigma=1.0)
    som.train(close_price_sequences, num_iteration=1000)

    # 3. Initialize Model
    print("Initializing NBEATSWithSOM model...")
    feature_extractor = SOMFeatureExtractor(som)
    model = NBEATSWithSOM(
        som_feature_extractor=feature_extractor,
        sequence_length=30,
        forecast_horizon=5,
        n_stacks=2,
        n_blocks=3,
        n_layers=4,
        layer_width=256,
        num_features=len(scaled_feature_cols)
    )

    # 4. Train the NBEATS Model
    print("Training NBEATS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 30
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss / len(train_loader):.4f}")

    # 5. Evaluate the Model
    print("Evaluating model...")
    model.eval()
    all_y_true_scaled = []
    all_y_pred_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            all_y_true_scaled.append(y_batch.cpu().numpy())
            all_y_pred_scaled.append(y_pred.cpu().numpy())

    y_true_scaled_np = np.concatenate(all_y_true_scaled)
    y_pred_scaled_np = np.concatenate(all_y_pred_scaled)

    # --- Inverse Transform Predictions using the target_scaler ---
    y_true_np = target_scaler.inverse_transform(y_true_scaled_np)
    y_pred_np = target_scaler.inverse_transform(y_pred_scaled_np)

    # Calculate Metrics on original scale
    rmse = np.sqrt(np.mean((y_true_np - y_pred_np)**2))
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    dir_accuracy = calculate_directional_accuracy(y_true_np, y_pred_np)

    print("\n--- Performance Metrics (on original scale) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Directional Accuracy: {dir_accuracy:.2f}%")
    print("-------------------------------------------------\n")

if __name__ == "__main__":
    main()
