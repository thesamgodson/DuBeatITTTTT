import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.som_analysis.som_trainer import SimpleSOM


def load_artifacts(results_dir='results'):
    """Load the artifacts saved from the evaluation script."""
    print("Loading artifacts...")
    with open(os.path.join(results_dir, 'som_model.pkl'), 'rb') as f:
        som = pickle.load(f)
    with open(os.path.join(results_dir, 'test_data_and_winners.pkl'), 'rb') as f:
        data = pickle.load(f)
    print("Artifacts loaded.")
    return som, data

def calculate_heatmap_data(som, winner_map, data_to_map):
    """Calculate the average value of data_to_map for each neuron."""
    map_size = som.grid_size
    heatmap = np.zeros(map_size)
    counts = np.zeros(map_size)

    for i, winner in enumerate(winner_map):
        heatmap[winner] += data_to_map[i]
        counts[winner] += 1

    # Avoid division by zero for neurons that never won
    # Return NaN for unvisited neurons so they can be colored differently
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap /= counts
        heatmap[counts == 0] = np.nan

    return heatmap

def plot_u_matrix(som, save_path):
    """Plot the U-matrix of the SOM."""
    print("Plotting U-matrix...")
    u_matrix = som.som.distance_map()
    plt.figure(figsize=(10, 10))
    plt.pcolor(u_matrix.T, cmap='bone_r')
    plt.colorbar(label='Unified Distance')
    plt.title('SOM U-Matrix (Cluster Structure)')
    plt.xticks(np.arange(som.grid_size[0]))
    plt.yticks(np.arange(som.grid_size[1]))
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"U-matrix saved to {save_path}")

def plot_heatmap(heatmap_data, title, cmap='viridis', save_path=None, center_zero=False):
    """Plot a generic heatmap."""
    print(f"Plotting heatmap: {title}...")
    plt.figure(figsize=(10, 10))

    # Set background for NaN cells
    ax = plt.gca()
    ax.set_facecolor('gray')

    # Use a diverging colormap if we want to center on zero
    vmin = np.nanmin(heatmap_data)
    vmax = np.nanmax(heatmap_data)

    if center_zero and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plt.pcolor(heatmap_data.T, cmap=cmap, norm=norm, edgecolor='k', linewidths=0.5)
    else:
        plt.pcolor(heatmap_data.T, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='k', linewidths=0.5)

    plt.colorbar(label='Average Value')
    plt.title(title)
    plt.xticks(np.arange(heatmap_data.shape[0]))
    plt.yticks(np.arange(heatmap_data.shape[1]))
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to {save_path}")

def main():
    results_dir = 'results'
    som, data = load_artifacts(results_dir)

    X_test = data['X_test']
    y_test = data['y_test']
    winner_map = data['winner_map']

    # The saved feature names are the original names, not the scaled ones.
    # We need to find the index based on the original names.
    feature_names = data['feature_names']

    # 1. Plot U-Matrix
    plot_u_matrix(som, save_path=os.path.join(results_dir, 'som_u_matrix.png'))

    # 2. Plot Future Return Heatmap
    # We care about the average return over the forecast horizon
    avg_future_return = np.mean(y_test, axis=1)
    return_heatmap = calculate_heatmap_data(som, winner_map, avg_future_return)
    plot_heatmap(return_heatmap, 'Average Future Return per Neuron (Scaled)', cmap='coolwarm',
                 save_path=os.path.join(results_dir, 'heatmap_return.png'), center_zero=True)

    # 3. Plot Input Volatility Heatmap
    # Use the 'Close' price volatility (assuming it's the first feature)
    close_price_idx = feature_names.index('Close')
    input_volatility = np.std(X_test[:, :, close_price_idx], axis=1)
    volatility_heatmap = calculate_heatmap_data(som, winner_map, input_volatility)
    plot_heatmap(volatility_heatmap, 'Average Input Volatility per Neuron', cmap='plasma',
                 save_path=os.path.join(results_dir, 'heatmap_volatility.png'))

    # 4. Plot Input Volume Heatmap
    if 'Volume' in feature_names:
        volume_idx = feature_names.index('Volume')
        input_volume = np.mean(X_test[:, :, volume_idx], axis=1)
        volume_heatmap = calculate_heatmap_data(som, winner_map, input_volume)
        plot_heatmap(volume_heatmap, 'Average Input Volume per Neuron', cmap='viridis',
                     save_path=os.path.join(results_dir, 'heatmap_volume.png'))

if __name__ == "__main__":
    main()
