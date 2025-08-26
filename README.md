# Financial Time Series Forecasting with N-BEATS and SOM

This project explores the use of a novel neural network architecture, combining N-BEATS (Neural aBasis Expansion Analysis for Time Series) with Self-Organizing Maps (SOMs), to forecast the price direction of BTC-USD.

The goal is to create a model that provides a positive financial return when used as the basis for a trading strategy.

## How to Run the Evaluation

To replicate the results and evaluate the models, run the master evaluation script:

```bash
python scripts/run_full_evaluation.py
```

This will train each model configuration and print a detailed financial performance report.

## Model Performance Comparison

The following table summarizes the performance of the different model configurations based on the custom `FinancialMetricsEvaluator`. The evaluation was performed on BTC-USD data from 2020-01-01 to 2024-12-31.

| Model                       | Annual Return | Sharpe Ratio | Max Drawdown | Directional Accuracy |
| --------------------------- | ------------- | ------------ | ------------ | -------------------- |
| **NBEATS-SOM (Close Only)** | -59.83%       | -1.391       | 26.18%       | 47.63%               |
| **NBEATS-SOM (Close+Volume)** | -62.11%       | -1.443       | 26.18%       | 47.91%               |

### Interpretation of Results

As of the latest run, both models underperform a simple buy-and-hold strategy, yielding negative returns and Sharpe ratios. The inclusion of 'Volume' as a feature did not lead to a significant improvement in performance.

**Note on Metrics:** The evaluation script uses a comprehensive, custom-built financial metrics evaluator. Some of the reported metrics (e.g., "Win Rate") may produce counter-intuitive results and require further validation and refinement. The primary metrics for comparison are the Sharpe Ratio and Annual Return.
