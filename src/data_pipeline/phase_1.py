import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
import time
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data processing"""
    assets: List[str] = None
    start_date: str = '2015-01-01'
    end_date: str = '2024-12-31'
    pattern_windows: List[int] = None
    forecast_horizon: int = 1
    validation_split: float = 0.2
    test_split: float = 0.2

    def __post_init__(self):
        if self.assets is None:
            self.assets = ['BTC-USD']
        if self.pattern_windows is None:
            self.pattern_windows = [3, 5, 7, 10, 15]

class EnhancedDataLoader:
    """
    Enhanced data loader with hypergraph-specific preprocessing
    Supports multi-asset loading and pattern extraction
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.scalers = {}
        self.raw_data = {}

    def load_asset_data(self, asset: str) -> pd.DataFrame:
        """Load single asset data with error handling"""
        try:
            ticker = yf.Ticker(asset)
            df = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True
            )

            if df.empty:
                raise ValueError(f"No data retrieved for {asset}")

            # Reset index and clean
            df = df.reset_index()
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

            # Add basic features
            df['Returns'] = df['Close'].pct_change()
            df['LogReturns'] = np.log(df['Close']).diff()
            df['Volatility'] = df['Returns'].rolling(window=20).std()

            # Remove initial NaN values
            df = df.dropna()

            logger.info(f"Loaded {len(df)} records for {asset}")
            return df

        except Exception as e:
            logger.error(f"Failed to load {asset}: {e}")
            raise

    def load_all_assets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured assets"""
        data = {}

        for asset in self.config.assets:
            try:
                data[asset] = self.load_asset_data(asset)
            except Exception as e:
                logger.warning(f"Skipping {asset} due to error: {e}")

        if not data:
            raise ValueError("No assets successfully loaded")

        self.raw_data = data
        return data

    def validate_data_quality(self, df: pd.DataFrame, asset: str) -> bool:
        """Validate data quality for single asset"""
        issues = []

        # Check for missing values
        if df.isnull().any().any():
            issues.append("Contains NaN values")

        # Check for non-positive prices
        if (df['Close'] <= 0).any():
            issues.append("Contains non-positive prices")

        # Check for extreme outliers (>10 sigma)
        returns_zscore = np.abs(df['Returns'] / df['Returns'].std())
        if (returns_zscore > 10).any():
            issues.append("Contains extreme outliers")

        # Check temporal ordering
        if not df['Date'].is_monotonic_increasing:
            issues.append("Dates not properly ordered")

        # Check minimum length
        if len(df) < 100:
            issues.append("Insufficient data points")

        if issues:
            logger.warning(f"Data quality issues for {asset}: {issues}")
            return False

        logger.info(f"Data quality validation passed for {asset}")
        return True

class PatternExtractor:
    """
    Extract overlapping patterns from price sequences
    Supports multiple window sizes and normalization methods
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 7, 10]
        self.scalers = {}

    def extract_patterns_single_window(self,
                                     price_sequence: np.ndarray,
                                     window: int,
                                     normalize_method: str = 'z_score') -> np.ndarray:
        """
        Extract patterns for a single window size

        Args:
            price_sequence: 1D array of prices
            window: Pattern window size
            normalize_method: 'z_score', 'min_max', or 'returns'

        Returns:
            Array of shape (n_patterns, window) containing normalized patterns
        """
        if len(price_sequence) < window:
            return np.array([]).reshape(0, window)

        patterns = []

        for i in range(len(price_sequence) - window + 1):
            pattern = price_sequence[i:i + window].copy()

            # Apply normalization
            if normalize_method == 'z_score':
                if pattern.std() > 1e-8:  # Avoid division by zero
                    pattern = (pattern - pattern.mean()) / pattern.std()

            elif normalize_method == 'min_max':
                if pattern.max() != pattern.min():
                    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

            elif normalize_method == 'returns':
                if len(pattern) > 1:
                    pattern = np.diff(np.log(pattern + 1e-8))  # Log returns

            patterns.append(pattern)

        return np.array(patterns)

    def extract_all_patterns(self,
                           price_sequence: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract patterns for all configured window sizes"""
        all_patterns = {}

        for window in self.windows:
            patterns = self.extract_patterns_single_window(price_sequence, window)
            all_patterns[window] = patterns

        return all_patterns

    def validate_patterns(self, patterns_dict: Dict[int, np.ndarray]) -> bool:
        """Validate extracted patterns"""
        issues = []

        for window, patterns in patterns_dict.items():
            if len(patterns) == 0:
                issues.append(f"No patterns extracted for window {window}")
                continue

            # Check for NaN values
            if np.isnan(patterns).any():
                issues.append(f"NaN values in window {window} patterns")

            # Check for infinite values
            if np.isinf(patterns).any():
                issues.append(f"Infinite values in window {window} patterns")

            # Check pattern diversity (not all the same)
            if patterns.shape[0] > 1:
                pattern_variance = np.var(patterns, axis=0)
                if np.all(pattern_variance < 1e-8):
                    issues.append(f"No variance in window {window} patterns")

        if issues:
            logger.warning(f"Pattern validation issues: {issues}")
            return False

        return True

class TemporalAligner:
    """
    Ensure proper temporal alignment between patterns and targets
    Critical for preventing data leakage
    """

    def __init__(self, forecast_horizon: int = 1):
        self.forecast_horizon = forecast_horizon

    def create_aligned_dataset(self,
                             df: pd.DataFrame,
                             pattern_windows: List[int],
                             target_column: str = 'LogReturns') -> Dict:
        """
        Create temporally aligned dataset

        Args:
            df: DataFrame with price data
            pattern_windows: List of pattern window sizes
            target_column: Column name for prediction targets

        Returns:
            Dictionary with aligned sequences, patterns, and targets
        """
        max_window = max(pattern_windows)

        # Extract sequences for pattern creation
        price_sequences = []
        all_patterns = {window: [] for window in pattern_windows}
        targets = []
        timestamps = []

        extractor = PatternExtractor(pattern_windows)

        # Create sliding windows
        for i in range(max_window, len(df) - self.forecast_horizon):
            # Price sequence for pattern extraction
            price_seq = df['Close'].iloc[i-max_window:i].values
            price_sequences.append(price_seq)

            # Extract patterns for this sequence
            seq_patterns = extractor.extract_all_patterns(price_seq)

            for window in pattern_windows:
                if len(seq_patterns[window]) > 0:
                    # Take the most recent pattern
                    all_patterns[window].append(seq_patterns[window][-1])
                else:
                    # Fallback: use last 'window' prices directly
                    fallback_pattern = price_seq[-window:]
                    if len(fallback_pattern) == window:
                        all_patterns[window].append(fallback_pattern)

            # Future target (avoid data leakage)
            target = df[target_column].iloc[i + self.forecast_horizon - 1]
            targets.append(target)

            # Timestamp for reference
            timestamps.append(df['Date'].iloc[i])

        # Convert to numpy arrays
        result = {
            'sequences': np.array(price_sequences),
            'patterns': {w: np.array(patterns) for w, patterns in all_patterns.items()},
            'targets': np.array(targets),
            'timestamps': timestamps
        }

        return result

    def split_temporal_data(self,
                          aligned_data: Dict,
                          val_split: float = 0.2,
                          test_split: float = 0.2) -> Tuple[Dict, Dict, Dict]:
        """
        Split data temporally (no random shuffling to avoid data leakage)
        """
        total_samples = len(aligned_data['targets'])

        # Calculate split points
        train_end = int(total_samples * (1 - val_split - test_split))
        val_end = int(total_samples * (1 - test_split))

        def split_dict(data_dict, start_idx, end_idx):
            split_data = {}
            for key, value in data_dict.items():
                if key == 'patterns':
                    split_data[key] = {w: patterns[start_idx:end_idx]
                                     for w, patterns in value.items()}
                else:
                    split_data[key] = value[start_idx:end_idx]
            return split_data

        train_data = split_dict(aligned_data, 0, train_end)
        val_data = split_dict(aligned_data, train_end, val_end)
        test_data = split_dict(aligned_data, val_end, total_samples)

        logger.info(f"Data split: Train={len(train_data['targets'])}, "
                   f"Val={len(val_data['targets'])}, "
                   f"Test={len(test_data['targets'])}")

        return train_data, val_data, test_data

# Comprehensive testing suite
def test_phase1_implementation():
    """Comprehensive test suite for Phase 1"""

    print("="*60)
    print("PHASE 1 TESTING: Enhanced Data Pipeline")
    print("="*60)

    # Test 1: Basic data loading
    print("\n1. Testing basic data loading...")
    config = DataConfig(
        assets=['BTC-USD'],
        start_date='2020-01-01',
        end_date='2020-12-31',
        pattern_windows=[5, 7, 10]
    )

    loader = EnhancedDataLoader(config)
    try:
        data = loader.load_all_assets()
        btc_data = data['BTC-USD']

        # Validate data quality
        is_valid = loader.validate_data_quality(btc_data, 'BTC-USD')
        assert is_valid, "Data quality validation failed"

        print(f"✅ Loaded {len(btc_data)} BTC records")
        print(f"✅ Data quality validation passed")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

    # Test 2: Pattern extraction
    print("\n2. Testing pattern extraction...")
    try:
        extractor = PatternExtractor([5, 7, 10])
        price_seq = btc_data['Close'].values[:100]  # Use subset for testing

        patterns = extractor.extract_all_patterns(price_seq)
        is_valid_patterns = extractor.validate_patterns(patterns)

        assert is_valid_patterns, "Pattern validation failed"

        for window, pattern_array in patterns.items():
            print(f"✅ Window {window}: {len(pattern_array)} patterns extracted")

    except Exception as e:
        print(f"❌ Pattern extraction failed: {e}")
        return False

    # Test 3: Temporal alignment
    print("\n3. Testing temporal alignment...")
    try:
        aligner = TemporalAligner(forecast_horizon=1)
        aligned_data = aligner.create_aligned_dataset(
            btc_data,
            pattern_windows=[5, 7, 10],
            target_column='LogReturns'
        )

        # Check alignment integrity
        n_samples = len(aligned_data['targets'])
        assert n_samples > 0, "No aligned samples created"

        for window in [5, 7, 10]:
            assert len(aligned_data['patterns'][window]) == n_samples, \
                f"Misaligned patterns for window {window}"

        print(f"✅ Created {n_samples} aligned samples")

    except Exception as e:
        print(f"❌ Temporal alignment failed: {e}")
        return False

    # Test 4: Data splitting
    print("\n4. Testing data splitting...")
    try:
        train_data, val_data, test_data = aligner.split_temporal_data(aligned_data)

        # Check split integrity
        total_original = len(aligned_data['targets'])
        total_split = (len(train_data['targets']) +
                      len(val_data['targets']) +
                      len(test_data['targets']))

        assert total_original == total_split, "Data lost during splitting"

        print(f"✅ Train: {len(train_data['targets'])} samples")
        print(f"✅ Val: {len(val_data['targets'])} samples")
        print(f"✅ Test: {len(test_data['targets'])} samples")

    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return False

    print("\n" + "="*60)
    print("✅ PHASE 1 COMPLETE: All tests passed!")
    print("Ready to proceed to Phase 2: Hypergraph Builder")
    print("="*60)

    return True

# Performance benchmarks
def benchmark_phase1_performance():
    """Benchmark Phase 1 performance"""
    import time

    print("\nPerformance Benchmarks:")
    print("-" * 30)

    config = DataConfig(start_date='2015-01-01', end_date='2024-12-31')
    loader = EnhancedDataLoader(config)

    # Benchmark data loading
    start_time = time.time()
    data = loader.load_all_assets()
    load_time = time.time() - start_time

    print(f"Data Loading: {load_time:.2f} seconds")

    # Benchmark pattern extraction
    btc_data = data['BTC-USD']
    extractor = PatternExtractor([5, 7, 10, 15, 20])

    start_time = time.time()
    patterns = extractor.extract_all_patterns(btc_data['Close'].values)
    pattern_time = time.time() - start_time

    print(f"Pattern Extraction: {pattern_time:.2f} seconds")

    # Memory usage estimation
    total_size = sum(sys.getsizeof(p) for p in patterns.values())
    print(f"Pattern Memory Usage: {total_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # Run comprehensive tests
    success = test_phase1_implementation()

    if success:
        # Run performance benchmarks
        benchmark_phase1_performance()

        print("\n🎉 Phase 1 implementation complete and tested!")
        print("Next: Implement Phase 2 - Hypergraph Builder")
    else:
        print("\n❌ Phase 1 implementation failed")
        print("Fix issues before proceeding to Phase 2")
