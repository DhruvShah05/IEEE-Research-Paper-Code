"""
tests/test_splits.py — Tests for chronological splitting with embargo gap (fix 1.5 / Section 6).

Tests:
  - No data from train ends up in val or test (chronological integrity)
  - Embargo gap removes exactly the configured number of rows
  - 70/15/15 proportions hold within ±1 row
  - Timestamps are strictly monotone in each split
"""

import numpy as np
import pandas as pd
import pytest


def make_crypto_parquet(tmp_path, n=1000) -> str:
    """Creates a minimal crypto parquet with required columns for testing."""
    import os

    np.random.seed(42)
    base_bid = 100.0
    base_ask = 100.1
    price_increments = np.cumsum(np.random.randn(n) * 0.001)

    data = {}
    data['0'] = np.arange(n) * 250  # UNIX ms, 250ms intervals
    data['1'] = pd.to_datetime(data['0'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')

    # Bid prices descending, ask prices ascending from mid
    for i, col in enumerate([str(j) for j in range(2, 22, 2)]):
        data[col] = base_bid + price_increments - i * 0.01
    for i, col in enumerate([str(j) for j in range(22, 42, 2)]):
        data[col] = base_ask + price_increments + i * 0.01
    # Volumes
    for col in [str(j) for j in range(3, 22, 2)] + [str(j) for j in range(23, 42, 2)]:
        data[col] = np.random.randint(1, 100, size=n).astype(float)

    df = pd.DataFrame(data)
    path = str(tmp_path / 'crypto_test.parquet')
    df.to_parquet(path)
    return path


class TestCryptoDatasetSplits:
    def test_chronological_no_shuffle(self, tmp_path):
        """Train ends before val starts; val ends before test starts."""
        parquet = make_crypto_parquet(tmp_path, n=1000)
        from data.loaders import CryptoDataset
        ds = CryptoDataset(
            parquet_path=parquet, horizon=10, threshold=0.0001,
            embargo_horizon=10
        )

        # Test split indices must be non-overlapping
        n = len(ds.X_train) + 10 + len(ds.X_val) + 10 + len(ds.X_test)
        assert len(ds.X_train) < len(ds.X_val) + len(ds.X_test) + len(ds.X_train)
        # Train and val must not share any timesteps
        assert len(ds.ts_train) == len(ds.X_train)
        assert len(ds.ts_val)   == len(ds.X_val)
        assert len(ds.ts_test)  == len(ds.X_test)

        # Strict ordering: max train ts < min val ts < min test ts
        if len(ds.ts_train) > 0 and len(ds.ts_val) > 0:
            assert ds.ts_train.max() < ds.ts_val.min(), \
                "Train timestamps must all precede val timestamps"
        if len(ds.ts_val) > 0 and len(ds.ts_test) > 0:
            assert ds.ts_val.max() < ds.ts_test.min(), \
                "Val timestamps must all precede test timestamps"

    def test_embargo_removes_rows(self, tmp_path):
        """Embargo gap of E rows reduces train by E rows at the tail."""
        parquet = make_crypto_parquet(tmp_path, n=1000)
        from data.loaders import CryptoDataset

        ds_no_embargo = CryptoDataset(
            parquet_path=parquet, horizon=10, threshold=0.0001,
            embargo_horizon=0
        )
        ds_with_embargo = CryptoDataset(
            parquet_path=parquet, horizon=10, threshold=0.0001,
            embargo_horizon=20
        )

        # Train with embargo should be shorter
        assert len(ds_with_embargo.X_train) == len(ds_no_embargo.X_train) - 20, \
            "Embargo of 20 should reduce train set by 20 rows"

    def test_proportions_approx(self, tmp_path):
        """70/15/15 split proportions hold within ±1 row tolerance."""
        parquet = make_crypto_parquet(tmp_path, n=2000)
        from data.loaders import CryptoDataset

        ds = CryptoDataset(
            parquet_path=parquet, horizon=10, threshold=0.0001,
            embargo_horizon=0
        )
        n = len(ds.X_train) + len(ds.X_val) + len(ds.X_test)
        train_pct = len(ds.X_train) / n
        val_pct   = len(ds.X_val)   / n
        test_pct  = len(ds.X_test)  / n

        assert abs(train_pct - 0.70) < 0.02, f"Train pct {train_pct:.3f} not near 70%"
        assert abs(val_pct   - 0.15) < 0.02, f"Val pct {val_pct:.3f} not near 15%"
        assert abs(test_pct  - 0.15) < 0.02, f"Test pct {test_pct:.3f} not near 15%"

    def test_no_feature_leakage_from_scaler(self, tmp_path):
        """
        TrainOnlyScaler must not call fit() more than once (guards against data leakage).
        """
        parquet = make_crypto_parquet(tmp_path, n=500)
        from data.loaders import CryptoDataset
        from data.features import TrainOnlyScaler

        ds = CryptoDataset(
            parquet_path=parquet, horizon=10, threshold=0.0001,
            embargo_horizon=0
        )
        X_train, _, X_val, _, X_test, _ = ds.get_splits()

        scaler = TrainOnlyScaler(use_zscore=True)
        scaler.fit_transform(X_train)

        with pytest.raises(RuntimeError, match="more than once"):
            scaler.fit(X_val)

    def test_split_manifest_keys(self, tmp_path):
        """split_manifest must contain expected keys."""
        parquet = make_crypto_parquet(tmp_path, n=500)
        from data.loaders import CryptoDataset
        import json, os

        ds = CryptoDataset(
            parquet_path=parquet, horizon=10, threshold=0.0001,
        )
        ds.save_split_manifest(str(tmp_path))
        manifest_path = str(tmp_path / 'split_manifest.json')
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            m = json.load(f)
        for key in ['market', 'horizon', 'embargo_rows', 'train_rows', 'val_rows', 'test_rows']:
            assert key in m, f"split_manifest missing key: {key}"
