import numpy as np
from collections import defaultdict

def analyze_som_patterns(som, patterns, future_returns):
    """
    Analyzes the predictive power of a trained SOM by mapping patterns to winning
    neurons and calculating the average future return for each neuron.

    Args:
        som (SimpleSOM): An instance of a trained SimpleSOM.
        patterns (np.ndarray): The input patterns used for training/analysis.
        future_returns (np.ndarray or pd.Series): The future returns corresponding
                                                   to each pattern.

    Returns:
        dict: A dictionary where keys are neuron coordinates (tuples) and
              values are the average future return for that neuron.
    """
    if len(patterns) != len(future_returns):
        raise ValueError("The number of patterns and future_returns must be the same.")

    # A dictionary to store the returns for each winning neuron.
    # The key is the neuron's (x, y) coordinate, value is a list of returns.
    neuron_map = defaultdict(list)

    # Map each pattern to its winner and store the corresponding future return.
    for pattern, future_return in zip(patterns, future_returns):
        winner_neuron = som.winner(pattern)
        neuron_map[winner_neuron].append(future_return)

    # Calculate the average return for each neuron.
    # If a neuron was never a winner, it won't be in the dictionary.
    average_return_map = {
        neuron: np.mean(returns) for neuron, returns in neuron_map.items()
    }

    return average_return_map
