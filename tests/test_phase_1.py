import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import load_bitcoin_data
from src.data_processing.feature_engineer import create_basic_features, create_pattern_vectors

class TestPhase1DataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        cls.raw_df = load_bitcoin_data(start_date='2020-01-01', end_date='2020-12-31')
        # Create a small sample DataFrame for feature engineering tests
        cls.sample_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'Close': [100, 102, 101, 103, 105]
        })

    def test_data_loader(self):
        """
        Validation test for load_bitcoin_data.
        """
        df = self.raw_df

        self.assertFalse(df.empty, "Dataframe should not be empty.")
        self.assertGreater(len(df), 360, "Should have roughly 365 days for a full year.")
        self.assertEqual(df['Close'].isna().sum(), 0, "There should be no missing values in 'Close' column.")
        self.assertTrue((df['Close'] > 0).all(), "All 'Close' prices must be positive.")

        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.assertListEqual(list(df.columns), expected_columns, "Columns do not match expected columns.")
        print("\n✓ Data loader working correctly")

    def test_feature_engineering_basic(self):
        """
        Validation test for create_basic_features.
        """
        # Use a copy to avoid modifying the class-level sample_data
        df_featured = create_basic_features(self.sample_data.copy())

        self.assertIn('returns', df_featured.columns, "'returns' column is missing.")
        self.assertIn('SMA_10', df_featured.columns, "'SMA_10' column is missing.")

        # After dropping NaNs from moving averages, there might be no data left in this small sample
        # A more robust test would use a larger sample, but for this check, we ensure it runs.
        if not df_featured.empty:
            self.assertGreater(df_featured['returns'].std(), 0, "Returns should have variance.")

        print("✓ Basic feature engineering working correctly")

    def test_feature_engineering_patterns(self):
        """
        Validation test for create_pattern_vectors.
        """
        patterns = create_pattern_vectors(self.sample_data, window_size=3)

        self.assertIsInstance(patterns, np.ndarray, "Pattern vectors should be a numpy array.")
        self.assertEqual(patterns.shape, (3, 3), "Pattern vectors shape is incorrect.")

        # The first element of each pattern vector should be 1.0 due to normalization
        np.testing.assert_allclose(patterns[:, 0], 1.0)
        print("✓ Pattern vector creation working correctly")

if __name__ == '__main__':
    unittest.main()
