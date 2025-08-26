import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple

class SanityCheckStrategies:
    """
    Test the most basic strategies to establish if your market/timeframe
    has any predictable patterns at all.
    """

    def __init__(self, symbol: str = "BTC-USD", start: str = "2020-01-01", end: str = "2024-12-31"):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = None

    def load_data(self):
        """Load and prepare basic data."""
        print(f"Loading {self.symbol} data from {self.start} to {self.end}...")
        self.data = yf.download(self.symbol, start=self.start, end=self.end, progress=False)
        self.data['returns'] = self.data['Close'].pct_change()
        self.data['next_return'] = self.data['returns'].shift(-1)
        self.data = self.data.dropna()
        print(f"Loaded {len(self.data)} trading days")

    def calculate_strategy_performance(self, signals: np.array, returns: np.array,
                                     name: str, transaction_cost: float = 0.001) -> Dict:
        """Calculate performance metrics for a strategy."""

        if len(signals) != len(returns):
            raise ValueError(f"Length mismatch for '{name}': signals ({len(signals)}) vs returns ({len(returns)})")

        # Ensure signals are -1, 0, 1
        if np.all(np.isin(signals, [0, 1])):
            signals = signals * 2 - 1

        strategy_returns = signals * returns

        # Manually calculate position changes to avoid numpy bugs with prepend
        position_changes = np.zeros_like(signals, dtype=bool)
        position_changes[0] = True # First day is always a trade
        position_changes[1:] = signals[1:] != signals[:-1]

        transaction_costs = position_changes * transaction_cost
        net_returns = strategy_returns - transaction_costs

        total_return = np.prod(1 + net_returns) - 1
        annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1 if len(net_returns) > 0 else 0
        annual_vol = np.std(net_returns) * np.sqrt(252)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        cumulative = np.cumprod(1 + net_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = abs(np.min(drawdown))

        trade_days = np.where(position_changes > 0)[0]
        winning_trades = np.sum(strategy_returns[trade_days] > 0)
        total_trades = len(trade_days)
        win_rate = winning_trades / max(total_trades, 1)

        return {
            'name': name,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_return': total_return
        }

    def test_all_sanity_strategies(self) -> pd.DataFrame:
        """Test all basic strategies and return results."""

        if self.data is None:
            self.load_data()

        returns = self.data['returns'].values
        next_returns = self.data['next_return'].values

        results = []

        # 1. Buy and Hold
        results.append(
            self.calculate_strategy_performance(np.ones_like(next_returns), next_returns, "Buy and Hold")
        )

        # 2. Random Strategy
        np.random.seed(42)
        results.append(
            self.calculate_strategy_performance(np.random.choice([-1, 1], len(next_returns)), next_returns, "Random")
        )

        # 3. Momentum (yesterday's return predicts today)
        momentum_signals = np.sign(returns)
        momentum_signals[momentum_signals == 0] = 1
        results.append(
            self.calculate_strategy_performance(momentum_signals, next_returns, "Momentum (1-day)")
        )

        # 4. Mean Reversion (opposite of yesterday)
        reversion_signals = -np.sign(returns)
        reversion_signals[reversion_signals == 0] = 1
        results.append(
            self.calculate_strategy_performance(reversion_signals, next_returns, "Mean Reversion (1-day)")
        )

        # 5. Moving Average Crossover (20-day)
        self.data['sma_20'] = self.data['Close'].rolling(20).mean().bfill()
        ma_signals = np.where(self.data['Close'].values > self.data['sma_20'].values, 1, -1)
        results.append(
            self.calculate_strategy_performance(ma_signals, next_returns, "MA(20) Crossover")
        )

        # 6. RSI Strategy
        self.data['rsi'] = self._calculate_rsi(self.data['Close']).bfill()
        rsi_signals = np.where(self.data['rsi'].values < 30, 1, np.where(self.data['rsi'].values > 70, -1, 0))
        rsi_signals = pd.Series(rsi_signals).replace(to_replace=0, method='ffill').values
        results.append(
            self.calculate_strategy_performance(rsi_signals, next_returns, "RSI(14) Strategy")
        )

        # 7. Volatility Strategy (buy low vol, sell high vol)
        self.data['vol_20'] = self.data['returns'].rolling(20).std().bfill()
        vol_signals = np.where(self.data['vol_20'].values < self.data['vol_20'].median(), 1, -1)
        results.append(
            self.calculate_strategy_performance(vol_signals, next_returns, "Low Volatility")
        )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        return results_df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def print_results(self, results_df: pd.DataFrame):
        print("\n" + "="*80)
        print("SANITY CHECK: BASIC STRATEGY PERFORMANCE")
        print("="*80)
        print(f"Market: {self.symbol} | Period: {self.start} to {self.end}")
        print("-"*80)

        for _, row in results_df.iterrows():
            print(f"{row['name']:<25} | "
                  f"Annual Return: {row['annual_return']:>7.2%} | "
                  f"Sharpe: {row['sharpe_ratio']:>6.2f} | "
                  f"Max DD: {row['max_drawdown']:>6.2%} | "
                  f"Win Rate: {row['win_rate']:>6.1%}")

        print("-"*80)
        print("INTERPRETATION:")

        best_strategy = results_df.iloc[0]
        worst_strategy = results_df.iloc[-1]

        print(f"🏆 Best Strategy: {best_strategy['name']} (Sharpe: {best_strategy['sharpe_ratio']:.2f})")
        print(f"💥 Worst Strategy: {worst_strategy['name']} (Sharpe: {worst_strategy['sharpe_ratio']:.2f})")

        buy_hold_sharpe = results_df[results_df['name'] == 'Buy and Hold']['sharpe_ratio'].iloc[0]

        if best_strategy['sharpe_ratio'] > 0.5:
            print("✅ Good news: Simple strategies can work in this market/timeframe")
        elif best_strategy['sharpe_ratio'] > 0:
            print("⚠️  Marginal: Simple strategies barely work")
        else:
            print("❌ Bad news: Even simple strategies fail")

        print(f"\n📊 NBEATS-SOM Models vs Benchmarks:")
        print(f"   Your NBEATS-SOM Sharpe: -1.39 to -1.44")
        print(f"   Buy and Hold Sharpe: {buy_hold_sharpe:.2f}")
        print(f"   Random Strategy Sharpe: {results_df[results_df['name'] == 'Random']['sharpe_ratio'].iloc[0]:.2f}")

        if buy_hold_sharpe > -1.0:
            print("🔍 CRITICAL INSIGHT: Your models are performing worse than random!")

if __name__ == "__main__":
    checker = SanityCheckStrategies()
    results = checker.test_all_sanity_strategies()
    checker.print_results(results)
