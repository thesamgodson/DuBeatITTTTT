import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.phase_1 import DataConfig, EnhancedDataLoader, PatternExtractor, TemporalAligner

class TestPhase1DataPipeline(unittest.TestCase):

    def test_phase1_implementation(self):
        """
        Comprehensive test suite for Phase 1, moved from the source file.
        """
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
        data = loader.load_all_assets()
        btc_data = data['BTC-USD']

        is_valid = loader.validate_data_quality(btc_data, 'BTC-USD')
        self.assertTrue(is_valid, "Data quality validation failed")

        print(f"✅ Loaded {len(btc_data)} BTC records")
        print(f"✅ Data quality validation passed")

        # Test 2: Pattern extraction
        print("\n2. Testing pattern extraction...")
        extractor = PatternExtractor([5, 7, 10])
        price_seq = btc_data['Close'].values[:100]

        patterns = extractor.extract_all_patterns(price_seq)
        is_valid_patterns = extractor.validate_patterns(patterns)

        self.assertTrue(is_valid_patterns, "Pattern validation failed")

        for window, pattern_array in patterns.items():
            print(f"✅ Window {window}: {len(pattern_array)} patterns extracted")

        # Test 3: Temporal alignment
        print("\n3. Testing temporal alignment...")
        aligner = TemporalAligner(forecast_horizon=1)
        aligned_data = aligner.create_aligned_dataset(
            btc_data,
            pattern_windows=[5, 7, 10],
            target_column='LogReturns'
        )

        n_samples = len(aligned_data['targets'])
        self.assertGreater(n_samples, 0, "No aligned samples created")

        for window in [5, 7, 10]:
            self.assertEqual(len(aligned_data['patterns'][window]), n_samples,
                             f"Misaligned patterns for window {window}")

        print(f"✅ Created {n_samples} aligned samples")

        # Test 4: Data splitting
        print("\n4. Testing data splitting...")
        train_data, val_data, test_data = aligner.split_temporal_data(aligned_data)

        total_original = len(aligned_data['targets'])
        total_split = (len(train_data['targets']) +
                       len(val_data['targets']) +
                       len(test_data['targets']))

        self.assertEqual(total_original, total_split, "Data lost during splitting")

        print(f"✅ Train: {len(train_data['targets'])} samples")
        print(f"✅ Val: {len(val_data['targets'])} samples")
        print(f"✅ Test: {len(test_data['targets'])} samples")

        print("\n" + "="*60)
        print("✅ PHASE 1 COMPLETE: All tests passed!")
        print("Ready to proceed to Phase 2: Hypergraph Builder")
        print("="*60)

if __name__ == '__main__':
    unittest.main()
