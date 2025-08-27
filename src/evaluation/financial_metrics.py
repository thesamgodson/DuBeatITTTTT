import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

class FinancialMetricsEvaluator:
    """
    Proper evaluation metrics for financial forecasting models.
    Focuses on economic value rather than statistical accuracy.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize with risk-free rate (annualized).
        Default is 2% (typical for developed markets).
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(self,
                            predictions: np.array,
                            actual_returns: np.array,
                            actual_prices: Optional[np.array] = None,
                            transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate comprehensive financial metrics.

        Args:
            predictions: Predicted directions (1 for up, 0 for down) or returns
            actual_returns: Actual returns (not prices)
            actual_prices: Optional, for price-based metrics
            transaction_cost: Transaction cost per trade (0.1% default)

        Returns:
            Dictionary of all relevant metrics
        """

        # Ensure arrays are numpy arrays
        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

        # Remove any NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
        if len(predictions.shape) > 1:
            mask = mask.all(axis=1)

        predictions = predictions[mask]
        actual_returns = actual_returns[mask]

        if len(predictions) == 0:
            return self._empty_metrics_dict()

        # Calculate different types of metrics
        trading_metrics = self._calculate_trading_metrics(
            predictions, actual_returns, transaction_cost
        )

        risk_metrics = self._calculate_risk_metrics(actual_returns)

        accuracy_metrics = self._calculate_accuracy_metrics(
            predictions, actual_returns
        )

        economic_metrics = self._calculate_economic_metrics(
            predictions, actual_returns, transaction_cost
        )

        # Combine all metrics
        all_metrics = {
            **trading_metrics,
            **risk_metrics,
            **accuracy_metrics,
            **economic_metrics
        }

        return all_metrics

    def _calculate_trading_metrics(self, predictions: np.array,
                                 actual_returns: np.array,
                                 transaction_cost: float) -> Dict[str, float]:
        """Calculate trading-specific performance metrics."""

        # Convert predictions to trading signals
        if np.all(np.isin(predictions, [0, 1])):
            # Binary predictions: convert to -1, 1
            signals = predictions * 2 - 1  # 0,1 -> -1,1
        else:
            # Continuous predictions: use sign
            signals = np.sign(predictions)

        # Calculate strategy returns (before transaction costs)
        strategy_returns = signals * actual_returns

        # Apply transaction costs
        position_changes = np.abs(np.diff(signals, prepend=signals[0]))
        transaction_costs = position_changes * transaction_cost
        net_strategy_returns = strategy_returns - transaction_costs

        # Calculate metrics
        total_trades = np.sum(position_changes > 0)
        winning_trades = np.sum(strategy_returns[position_changes > 0] > 0)
        losing_trades = np.sum(strategy_returns[position_changes > 0] < 0)

        avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.sum(strategy_returns > 0) > 0 else 0
        avg_loss = np.mean(strategy_returns[strategy_returns < 0]) if np.sum(strategy_returns < 0) > 0 else 0

        return {
            'total_trades': float(total_trades),
            'winning_trades': float(winning_trades),
            'losing_trades': float(losing_trades),
            'win_rate': float(winning_trades / max(total_trades, 1)),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'win_loss_ratio': float(abs(avg_win / avg_loss)) if avg_loss != 0 else np.inf,
            'profit_factor': float(np.sum(strategy_returns[strategy_returns > 0]) /
                                 abs(np.sum(strategy_returns[strategy_returns < 0])))
                                 if np.sum(strategy_returns < 0) > 0 else np.inf
        }

    def _calculate_risk_metrics(self, returns: np.array) -> Dict[str, float]:
        """Calculate risk-based metrics."""

        if len(returns) == 0:
            return {}

        # Annualize assuming daily returns
        annual_factor = np.sqrt(252)

        volatility = np.std(returns) * annual_factor

        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        return {
            'volatility_annualized': float(volatility),
            'max_drawdown': float(abs(max_drawdown)),
            'var_95': float(var_95),
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurt())
        }

    def _calculate_accuracy_metrics(self, predictions: np.array,
                                  actual_returns: np.array) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""

        # Directional accuracy
        if np.all(np.isin(predictions, [0, 1])):
            pred_direction = predictions
        else:
            pred_direction = (predictions > 0).astype(int)

        actual_direction = (actual_returns > 0).astype(int)
        directional_accuracy = np.mean(pred_direction == actual_direction)

        # Mean Absolute Error (if predictions are returns)
        if not np.all(np.isin(predictions, [0, 1])):
            mae = np.mean(np.abs(predictions - actual_returns))
            rmse = np.sqrt(np.mean((predictions - actual_returns) ** 2))
        else:
            mae = np.nan
            rmse = np.nan

        return {
            'directional_accuracy': float(directional_accuracy),
            'mae': float(mae) if not np.isnan(mae) else None,
            'rmse': float(rmse) if not np.isnan(rmse) else None
        }

    def _calculate_economic_metrics(self, predictions: np.array,
                                  actual_returns: np.array,
                                  transaction_cost: float) -> Dict[str, float]:
        """Calculate economic value metrics - the most important ones."""

        if len(predictions) == 0:
            return {}

        # Convert to trading signals
        if np.all(np.isin(predictions, [0, 1])):
            signals = predictions * 2 - 1
        else:
            signals = np.sign(predictions)

        strategy_returns = signals * actual_returns

        position_changes = np.abs(np.diff(signals, prepend=signals[0]))
        transaction_costs = position_changes * transaction_cost
        net_returns = strategy_returns - transaction_costs

        annual_return = np.mean(net_returns) * 252
        annual_vol = np.std(net_returns) * np.sqrt(252)

        excess_return = annual_return - self.risk_free_rate
        sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0

        benchmark_return = np.mean(actual_returns) * 252
        active_return = annual_return - benchmark_return
        tracking_error = np.std(net_returns - actual_returns) * np.sqrt(252)
        info_ratio = active_return / tracking_error if tracking_error > 0 else 0

        cumulative = np.cumprod(1 + net_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        calmar_ratio = annual_return / max_dd if max_dd > 0 else 0

        return {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'information_ratio': float(info_ratio),
            'calmar_ratio': float(calmar_ratio),
            'benchmark_return': float(benchmark_return),
            'active_return': float(active_return)
        }

    def _empty_metrics_dict(self) -> Dict[str, float]:
        """Return empty metrics dict for edge cases."""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'directional_accuracy': 0, 'sharpe_ratio': 0,
            'annual_return': 0, 'max_drawdown': 0
        }

    def print_evaluation_report(self, metrics: Dict[str, float]) -> None:
        """Print a comprehensive evaluation report."""
        print("\n" + "="*60)
        print("FINANCIAL FORECASTING EVALUATION REPORT")
        print("="*60)
        # ... (rest of the print function)
