import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance_heatmap(log_path: str, save_path: str):
    """
    Loads the feature importance log and plots a heatmap of average importance
    for each expert.
    """
    print("Generating feature importance heatmap...")
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}. Skipping heatmap.")
        return

    df = pd.read_csv(log_path)

    # Pivot table to get average importance for each expert and feature
    pivot = df.pivot_table(index='expert_id', columns='feature', values='importance', aggfunc='mean')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title('Average Feature Importance per Expert')
    plt.xlabel('Feature')
    plt.ylabel('Expert ID')
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance heatmap saved to {save_path}")

def plot_sharpe_ratio_timeline(results_log: list, save_path: str):
    """
    Plots the Sharpe Ratio for each rolling window to show performance over time.
    (This is a placeholder as the full results log is not saved yet)
    """
    print("Generating Sharpe Ratio timeline... (Placeholder)")
    # This function would need the full results log with dates and sharpe ratios.
    # For now, we'll create a placeholder image.
    plt.figure(figsize=(15, 7))
    plt.text(0.5, 0.5, 'Sharpe Ratio Timeline (Not Implemented)', horizontalalignment='center', verticalalignment='center', fontsize=16)
    plt.title('Performance Over Time')
    plt.savefig(save_path)
    plt.close()
    print(f"Placeholder regime timeline saved to {save_path}")


def main():
    os.makedirs('results', exist_ok=True)

    # Generate heatmap from the feature importance log
    plot_feature_importance_heatmap(
        log_path='results/expert_feature_importance.csv',
        save_path='results/feature_importance_heatmap.png'
    )

    # Generate a placeholder for the regime timeline
    # In a real run, this would take the detailed results log
    plot_sharpe_ratio_timeline(
        results_log=[],
        save_path='results/regime_timeline.png'
    )

    # UMAP visualization is skipped as no new nodes were created.

if __name__ == "__main__":
    main()
