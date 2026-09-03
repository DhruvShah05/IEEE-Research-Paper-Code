"""
data/loaders.py — Authoritative data-loading interface for both markets.

build.md §6 specification:
  FI2010Dataset  : loads prepared .npy files, applies chronological 7/3-day split (train/test),
                   carves val from training days only, exposes horizon_k as attribute,
                   raises on missing columns.
  CryptoDataset  : loads prepared parquet, computes mid-price + labels from RAW columns
                   (fix 0.1), applies timestamp-based window slicing (fix 0.6),
                   performs 70/15/15 chronological split with embargo gap (1.5),
                   raises on missing columns.

Both return (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays.
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-coded FI-2010 file names (fix 0.4 — no glob)
# ---------------------------------------------------------------------------
# Official FI-2010 release: Ntakaris et al. (2018)
# Train: CF_7 (first 7 days)
# Test : CF_7, CF_8, CF_9 (days 7, 8, 9 of the 10-day recording)

_FI2010_TRAIN_FILES = [
    'Train_Dst_NoAuction_ZScore_CF_7.txt',
]
_FI2010_TEST_FILES = [
    'Test_Dst_NoAuction_ZScore_CF_7.txt',
    'Test_Dst_NoAuction_ZScore_CF_8.txt',
    'Test_Dst_NoAuction_ZScore_CF_9.txt',
]

# Expected row counts from the standard release (Ntakaris et al. 2018)
_FI2010_TRAIN_ROWS_EXPECTED = 254_750
_FI2010_TEST_ROWS_EXPECTED  = 139_587


# ---------------------------------------------------------------------------
# FI-2010 Dataset
# ---------------------------------------------------------------------------

class FI2010Dataset:
    """
    Loads FI-2010 prepared .npy files and performs the standard chronological
    7-days-train / 3-days-test split (using the pre-split Training / Testing
    directories, which match this convention in the official release).

    Fix 0.4: uses hard-coded file names instead of glob to avoid concatenating
    overlapping files (CF_1..CF_9 are cumulative, not independent).
    Fix 0.2: remaps official labels 1=Up,2=Stat,3=Down → 0=Down,1=Stat,2=Up.

    Attributes
    ----------
    horizon_k : int
        The prediction horizon used (k ∈ {10, 20, 30, 50, 100}).
    """

    # FI-2010: 144 feature columns + 5 label columns (one per horizon)
    N_FEATURES = 144
    HORIZON_TO_COL = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
    TOTAL_COLS = 149  # 144 features + 5 labels

    def __init__(
        self,
        train_path: str = 'data/processed/fi2010_train.npy',
        test_path: str = 'data/processed/fi2010_test.npy',
        horizon_k: int = 10,
        val_fraction: float = 0.2,
        feature_set: str = 'full144',
        embargo_horizon: int = None,
    ):
        """
        Parameters
        ----------
        train_path    : path to fi2010_train.npy  (produced by scripts/prepare_fi2010.py)
        test_path     : path to fi2010_test.npy
        horizon_k     : prediction horizon in {10, 20, 30, 50, 100}
        val_fraction  : fraction of training rows to carve off as validation (chronologically last)
        feature_set   : 'full144' (default) or 'raw40' (first 40 raw LOB columns only,
                        for fair feature-set comparison with crypto — 1.3 requirement)
        embargo_horizon : rows to drop at split boundaries (purge/embargo gap — 1.5).
                          Defaults to horizon_k if not set.
        """
        if horizon_k not in self.HORIZON_TO_COL:
            raise ValueError(
                f"horizon_k must be one of {sorted(self.HORIZON_TO_COL.keys())}, got {horizon_k}"
            )
        if feature_set not in ('full144', 'raw40'):
            raise ValueError(f"feature_set must be 'full144' or 'raw40', got {feature_set!r}")

        self.horizon_k = horizon_k
        self.feature_set = feature_set
        n_features = 40 if feature_set == 'raw40' else self.N_FEATURES
        embargo = embargo_horizon if embargo_horizon is not None else horizon_k

        for path in (train_path, test_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FI-2010 file not found: {path}\n"
                    "Run `python3 scripts/prepare_fi2010.py` first."
                )

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        # Column validation — fail loudly (build.md §6)
        for name, arr in [('fi2010_train', train_data), ('fi2010_test', test_data)]:
            if arr.shape[1] < self.TOTAL_COLS:
                raise ValueError(
                    f"{name}: expected at least {self.TOTAL_COLS} columns "
                    f"(144 features + 5 labels), got {arr.shape[1]}"
                )

        # Row-count assertions (fix 0.4)
        if train_data.shape[0] != _FI2010_TRAIN_ROWS_EXPECTED:
            logger.warning(
                f"FI-2010 train rows: {train_data.shape[0]:,} "
                f"(expected {_FI2010_TRAIN_ROWS_EXPECTED:,}). "
                "Check that only CF_7 training file is loaded."
            )
        if test_data.shape[0] != _FI2010_TEST_ROWS_EXPECTED:
            logger.warning(
                f"FI-2010 test rows: {test_data.shape[0]:,} "
                f"(expected {_FI2010_TEST_ROWS_EXPECTED:,}). "
                "Check that only CF_7/CF_8/CF_9 test files are loaded."
            )

        label_col = self.HORIZON_TO_COL[horizon_k]
        X_all_train = train_data[:, :n_features]

        # Fix 0.2: remap official labels 1=Up,2=Stat,3=Down → 0=Down,1=Stat,2=Up
        from data.labeling import remap_fi2010_labels
        y_all_train = remap_fi2010_labels(train_data[:, label_col].astype(int))

        X_test = test_data[:, :n_features]
        y_test  = remap_fi2010_labels(test_data[:, label_col].astype(int))

        # Carve validation from the END of training days (chronologically)
        split_idx = int(len(X_all_train) * (1.0 - val_fraction))

        # Purge/embargo gap at the train/val boundary (1.5)
        self.X_train = X_all_train[:split_idx - embargo]
        self.y_train = y_all_train[:split_idx - embargo]
        self.X_val   = X_all_train[split_idx:]
        self.y_val   = y_all_train[split_idx:]
        self.X_test  = X_test
        self.y_test  = y_test

        logger.info(
            f"FI-2010 loaded: train={len(self.X_train):,}  val={len(self.X_val):,}  "
            f"test={len(self.X_test):,}  horizon_k={horizon_k}  feature_set={feature_set}"
        )

        # Save split manifest (1.5)
        self._split_manifest = {
            'market': 'fi2010',
            'horizon_k': horizon_k,
            'feature_set': feature_set,
            'train_rows': len(self.X_train),
            'val_rows': len(self.X_val),
            'test_rows': len(self.X_test),
            'embargo_rows': embargo,
        }

    def get_splits(self):
        """Returns (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays."""
        return (
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            self.X_test,  self.y_test,
        )

    def save_split_manifest(self, out_dir: str) -> None:
        """Saves split_manifest.json to out_dir (1.5 requirement)."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'split_manifest.json')
        with open(path, 'w') as f:
            json.dump(self._split_manifest, f, indent=4)
        logger.info(f"Split manifest saved to {path}")


# ---------------------------------------------------------------------------
# Crypto Dataset
# ---------------------------------------------------------------------------

class CryptoDataset:
    """
    Loads the prepared crypto parquet, computes mid-price and labels from the
    **raw** price columns *before* calling ``to_relative_price`` (fix 0.1),
    applies timestamp-based window slicing (fix 0.6), enforces a purge/embargo
    gap at split boundaries (1.5), and returns a chronological 70/15/15
    train/val/test split.

    Raises a clear error if expected columns are absent (build.md §6).
    """

    # Expected feature columns: bid/ask prices and volumes for 10 levels
    FEATURE_COLS = [str(i) for i in range(2, 42)]
    REQUIRED_COLS = {'0', '1', '2', '22'}  # timestamp cols + best bid/ask

    def __init__(
        self,
        parquet_path: str = 'data/processed/crypto_data.parquet',
        horizon: int = 40,
        threshold: float = 0.0001,
        window_days=None,
        price_mode: str = 'fractional',
        volume_transform: str = 'log1p',
        labeling_scheme: str = 'point_to_point',
        embargo_horizon: int = None,
    ):
        """
        Parameters
        ----------
        parquet_path      : path produced by scripts/prepare_crypto.py
        horizon           : number of events ahead for the label
        threshold         : ±fractional return threshold for Up/Down classification
        window_days       : optional [start_ts, end_ts] as UNIX ms (fix 0.6 —
                            timestamp-based, not row-count based).  Also accepts
                            [start_day_int, end_day_int] for backward compat
                            (interpreted as day indices, converted internally).
        price_mode        : passed to to_relative_price() — 'fractional' (default),
                            'absolute_diff', or 'tick_units'
        volume_transform  : 'log1p' (default) or 'none'
        labeling_scheme   : 'point_to_point' (default) or 'smoothed_mean' (FI-2010-style)
        embargo_horizon   : rows to purge at split boundaries.  Defaults to ``horizon``.
        """
        import pandas as pd
        from data.labeling import apply_horizon_labeling
        from data.features import to_relative_price, apply_volume_transform

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Crypto parquet not found: {parquet_path}\n"
                "Run `python3 scripts/prepare_crypto.py` first."
            )

        df = pd.read_parquet(parquet_path)

        # Column validation — fail loudly (build.md §6)
        missing = self.REQUIRED_COLS - set(df.columns.astype(str))
        if missing:
            raise ValueError(
                f"Crypto parquet is missing required columns: {sorted(missing)}. "
                "Verify the source data matches the schema in build.md §2.2."
            )

        # Fix 0.6: Timestamp-based window slicing
        if window_days is not None:
            start_val, end_val = window_days
            ts_col = '0'  # UNIX ms column
            if isinstance(start_val, int) and start_val < 1_000_000_000_000:
                # Treat as day indices (backward compat): convert to timestamps
                min_ts = df[ts_col].min()
                day_ms = 24 * 3600 * 1000
                start_ts = min_ts + (start_val - 1) * day_ms
                end_ts   = min_ts + end_val * day_ms
                logger.warning(
                    "window_days interpreted as day indices. Prefer passing "
                    "UNIX ms timestamps directly (fix 0.6)."
                )
            else:
                start_ts, end_ts = start_val, end_val

            mask = (df[ts_col] >= start_ts) & (df[ts_col] < end_ts)
            df = df[mask].copy()
            logger.info(
                f"Crypto: window filtered — {len(df):,} rows "
                f"({pd.Timestamp(start_ts, unit='ms')} → {pd.Timestamp(end_ts, unit='ms')})"
            )

        # Fix 0.1: Compute mid-price from RAW columns BEFORE any transform
        raw_mid_price = (df['2'].astype(float) + df['22'].astype(float)) / 2.0

        # Save timestamps and mid_prices for the backtest (3.3 requirement)
        self.raw_timestamps = df['0'].values
        self.raw_mid_prices = raw_mid_price.values

        logger.info(
            f"Crypto: {len(df):,} rows loaded. Applying '{price_mode}' price transform..."
        )

        df_transformed = to_relative_price(df, mode=price_mode)
        df_transformed = apply_volume_transform(df_transformed, mode=volume_transform)

        # Fix 0.1: pass raw mid_price so labeling uses real prices
        embargo = embargo_horizon if embargo_horizon is not None else horizon
        X, y, returns = apply_horizon_labeling(
            df_transformed, horizon, threshold,
            mid_price=raw_mid_price, scheme=labeling_scheme
        )

        # Chronological 70/15/15 split — no shuffling (build.md §6)
        n = len(X)
        tr  = int(n * 0.70)
        val = int(n * 0.85)

        # Purge/embargo gap at split boundaries (1.5)
        self.X_train = X[:tr - embargo]
        self.y_train = y[:tr - embargo]
        self.X_val   = X[tr:val - embargo]
        self.y_val   = y[tr:val - embargo]
        self.X_test  = X[val:]
        self.y_test  = y[val:]

        # Matching slices for timestamps, mid_prices, returns
        self.ts_train  = self.raw_timestamps[:tr - embargo]
        self.ts_val    = self.raw_timestamps[tr:val - embargo]
        self.ts_test   = self.raw_timestamps[val:val + len(self.X_test)]

        self.mid_train = self.raw_mid_prices[:tr - embargo]
        self.mid_val   = self.raw_mid_prices[tr:val - embargo]
        self.mid_test  = self.raw_mid_prices[val:val + len(self.X_test)]

        self.ret_train = returns[:tr - embargo]
        self.ret_val   = returns[tr:val - embargo]
        self.ret_test  = returns[val:val + len(self.X_test)]

        from data.labeling import get_class_distribution
        self._split_manifest = {
            'market': 'crypto',
            'horizon': horizon,
            'threshold': threshold,
            'price_mode': price_mode,
            'volume_transform': volume_transform,
            'labeling_scheme': labeling_scheme,
            'embargo_rows': embargo,
            'train_rows': len(self.X_train),
            'val_rows': len(self.X_val),
            'test_rows': len(self.X_test),
            'train_class_dist': get_class_distribution(self.y_train),
            'val_class_dist': get_class_distribution(self.y_val),
            'test_class_dist': get_class_distribution(self.y_test),
        }

        logger.info(
            f"Crypto split: train={len(self.X_train):,}  val={len(self.X_val):,}  "
            f"test={len(self.X_test):,}  embargo={embargo}"
        )

    def get_splits(self):
        """Returns (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays."""
        return (
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            self.X_test,  self.y_test,
        )

    def save_split_manifest(self, out_dir: str) -> None:
        """Saves split_manifest.json to out_dir (1.5 requirement)."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'split_manifest.json')
        with open(path, 'w') as f:
            json.dump(self._split_manifest, f, indent=4)
        logger.info(f"Split manifest saved to {path}")


# ---------------------------------------------------------------------------
# Windowed Dataset (1.5) — for real DeepLOB and temporal Transformers
# ---------------------------------------------------------------------------

class WindowedDataset:
    """
    PyTorch-compatible Dataset that returns ``(T, F)`` sequences (the last T
    snapshots for each sample) instead of single snapshots.

    Required by the real DeepLOB (§2.1) and temporal Transformer (§2.2).
    The first ``T−1`` samples of each split are dropped so that no window
    crosses a split boundary.

    Usage
    -----
    >>> ds = WindowedDataset(X_train, y_train, T=100)
    >>> loader = DataLoader(ds, batch_size=256, shuffle=True)
    >>> x, y = next(iter(loader))  # x: (256, 100, 40)  y: (256,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, T: int = 100):
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, F)
        y : np.ndarray, shape (N,)
        T : int — sequence length (number of snapshots per sample)
        """
        import torch
        from torch.utils.data import Dataset

        class _Inner(Dataset):
            def __init__(self, X, y, T):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.long)
                self.T = T
                self.offset = T - 1  # first valid sample index

            def __len__(self):
                return len(self.y) - self.offset

            def __getitem__(self, idx):
                i = idx + self.offset
                window = self.X[i - self.T + 1: i + 1]   # shape (T, F)
                return window, self.y[i]

        self._dataset = _Inner(X, y, T)
        self.T = T
        self.n_valid = len(self._dataset)

    def __len__(self):
        return self.n_valid

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        """Underlying PyTorch Dataset, for use with DataLoader."""
        return self._dataset


# ---------------------------------------------------------------------------
# Multi-asset Crypto Dataset (1.5)
# ---------------------------------------------------------------------------

class MultiAssetCryptoDataset:
    """
    Loads crypto data for multiple (exchange, symbol) pairs using the same
    pipeline.  Returns a dict mapping ``"<exchange>_<symbol>"`` to the
    corresponding CryptoDataset instance.
    """

    def __init__(
        self,
        asset_configs: list,
        horizon: int = 40,
        threshold: float = 0.0001,
        **kwargs,
    ):
        """
        Parameters
        ----------
        asset_configs : list of dicts, each with keys:
            'parquet_path', 'exchange', 'symbol'
        horizon, threshold, **kwargs : forwarded to CryptoDataset
        """
        self.datasets = {}
        for cfg in asset_configs:
            key = f"{cfg['exchange']}_{cfg['symbol']}"
            logger.info(f"Loading {key}...")
            self.datasets[key] = CryptoDataset(
                parquet_path=cfg['parquet_path'],
                horizon=horizon,
                threshold=threshold,
                **kwargs,
            )

    def keys(self):
        return self.datasets.keys()

    def __getitem__(self, key):
        return self.datasets[key]

    def items(self):
        return self.datasets.items()
