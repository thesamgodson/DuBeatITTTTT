# Financial Time Series Forecasting with N-BEATS and SOM

This project explores the use of a novel neural network architecture, combining N-BEATS (Neural aBasis Expansion Analysis for Time Series) with Self-Organizing Maps (SOMs), to forecast the price direction of BTC-USD.

The goal is to create a model that provides a positive financial return when used as the basis for a trading strategy.

## How to Run the Evaluation

To replicate the results and evaluate the models, run the master evaluation script:
```bash
python scripts/run_full_evaluation.py
```

To run the baseline strategy sanity checks, run:
```bash
python scripts/run_sanity_checks.py
```

## Sanity Checks and Baseline Performance

Before evaluating complex models, a sanity check was performed to test simple, well-known strategies on the BTC-USD dataset from 2020-01-01 to 2024-12-31.

| Strategy                 | Annual Return | Sharpe Ratio |
| ------------------------ | ------------- | ------------ |
| **Buy and Hold**         | 42.90%        | 0.77         |
| **Mean Reversion (1-day)** | -6.08%        | -0.15        |
| **Random**               | -16.33%       | -0.34        |
| **Momentum (1-day)**     | -39.02%       | -0.77        |

The strong performance of "Buy and Hold" indicates a clear upward trend in the market during this period, suggesting the market is not entirely random and should be beatable.

## Model Performance Comparison

The following table summarizes the performance of the different N-BEATS configurations.

| Model                         | Annual Return | Sharpe Ratio | Directional Accuracy | Key Finding                                   |
| ----------------------------- | ------------- | ------------ | -------------------- | --------------------------------------------- |
| **Pure N-BEATS (Close Only)** | -80.83%       | -1.863       | 49.30%               | Performs worse than random.                   |
| **NBEATS-SOM (Close Only)**   | -78.36%       | -1.807       | 44.01%               | SOM gating with one feature also fails.       |
| **NBEATS-SOM (Close+Volume)** | **+67.40%**   | **+1.473**   | **53.20%**           | **Highly profitable but training is unstable.** |

### Interpretation of Results and Key Findings

The evaluation led to a series of critical, and surprising, findings:

1.  **The Core Problem is Not Just the SOM:** The "Pure N-BEATS" model performed catastrophically, proving that the baseline N-BEATS architecture itself is not well-suited for this problem without modification. The initial hypothesis that the SOM was the sole cause of failure was incorrect.

2.  **The Combination is Key:** The most complex model, `NBEATS-SOM (Close+Volume)`, was the *only* successful model. This suggests that there is a valuable interaction between the SOM-based gating mechanism and the multi-feature input (price and volume). The SOM may be identifying market regimes where the relationship between price and volume shifts, and the gating mechanism correctly adjusts the model's focus.

3.  **Result Stability is a Major Concern:** The successful model's training run showed extremely large and unstable loss values before settling. The final excellent result could be a lucky fluke. It is crucial to test the stability and reproducibility of this result before proceeding.

### Next Steps

The immediate priority is to investigate the stability of the successful `NBEATS-SOM (Close + Volume)` model. The plan is to run the evaluation multiple times with different random seeds to see if the high Sharpe ratio is a consistent outcome or a one-time anomaly.
