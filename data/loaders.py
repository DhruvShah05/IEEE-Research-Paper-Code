"""
data/loaders.py — Authoritative data-loading interface for both markets.

build.md §6 specification:
  FI2010Dataset  : loads prepared .npy files, applies chronological 7/3-day split (train/test),
                   carves val from training days only, exposes horizon_k as attribute,
                   raises on missing columns.
  CryptoDataset  : loads prepared parquet, applies horizon/threshold labeling,
                   performs 70/15/15 chronological split, raises on missing columns.

Both return (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays.
"""

import numpy as np
import os


class FI2010Dataset:
    """
    Loads FI-2010 prepared .npy files and performs the standard chronological
    7-days-train / 3-days-test split (using the pre-split Training / Testing directories,
    which match this convention in the official release).

    Attributes
    ----------
    horizon_k : int
        The prediction horizon used (k ∈ {10, 20, 30, 50, 100}).  Exposed as an
        attribute so downstream code never has to silently guess (build.md §2.1).
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
    ):
        """
        Parameters
        ----------
        train_path    : path to fi2010_train.npy  (produced by scripts/prepare_fi2010.py)
        test_path     : path to fi2010_test.npy
        horizon_k     : prediction horizon in {10, 20, 30, 50, 100}; hard-coded in config (build.md §2.1)
        val_fraction  : fraction of training rows to carve off as validation (chronologically last)
        """
        if horizon_k not in self.HORIZON_TO_COL:
            raise ValueError(
                f"horizon_k must be one of {sorted(self.HORIZON_TO_COL.keys())}, got {horizon_k}"
            )
        self.horizon_k = horizon_k

        for path in (train_path, test_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FI-2010 file not found: {path}\n"
                    "Run `python3 scripts/prepare_fi2010.py` first."
                )

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        # Column validation — fail loudly rather than silently proceeding (build.md §6)
        for name, arr in [('fi2010_train', train_data), ('fi2010_test', test_data)]:
            if arr.shape[1] < self.TOTAL_COLS:
                raise ValueError(
                    f"{name}: expected at least {self.TOTAL_COLS} columns "
                    f"(144 features + 5 labels), got {arr.shape[1]}"
                )

        label_col = self.HORIZON_TO_COL[horizon_k]
        X_all_train = train_data[:, :self.N_FEATURES]
        y_all_train = train_data[:, label_col].astype(int) - 1  # 1/2/3 → 0/1/2

        X_test = test_data[:, :self.N_FEATURES]
        y_test = test_data[:, label_col].astype(int) - 1

        # Carve validation from the END of training days (chronologically) — not from test (build.md §6)
        split_idx = int(len(X_all_train) * (1.0 - val_fraction))
        self.X_train = X_all_train[:split_idx]
        self.y_train = y_all_train[:split_idx]
        self.X_val   = X_all_train[split_idx:]
        self.y_val   = y_all_train[split_idx:]
        self.X_test  = X_test
        self.y_test  = y_test

    def get_splits(self):
        """Returns (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays."""
        return (
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            self.X_test,  self.y_test,
        )


class CryptoDataset:
    """
    Loads the prepared crypto parquet, applies horizon/threshold labeling,
    and returns a chronological 70/15/15 train/val/test split.

    Raises a clear error if expected columns are absent (build.md §6).
    """

    # Expected feature columns: bid/ask prices and volumes for 10 levels
    # Cols 2–21 = bids, 22–41 = asks (build.md §2.2)
    FEATURE_COLS = [str(i) for i in range(2, 42)]
    REQUIRED_COLS = {'2', '22'}  # best bid price, best ask price (needed for mid-price + labeling)

    def __init__(
        self,
        parquet_path: str = 'data/processed/crypto_data.parquet',
        horizon: int = 40,
        threshold: float = 0.0001,
        window_days=None,
    ):
        """
        Parameters
        ----------
        parquet_path : path produced by scripts/prepare_crypto.py
        horizon      : number of events ahead for the label (resolved via threshold sweep — build.md §2.2)
        threshold    : ±fractional return threshold for Up/Down classification (default ±0.01%)
        window_days  : optional [start_day, end_day] to restrict to a contiguous time window
        """
        import pandas as pd
        from data.labeling import apply_horizon_labeling
        from data.features import to_relative_price

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Crypto parquet not found: {parquet_path}\n"
                "Run `python3 scripts/prepare_crypto.py` first."
            )

        df = pd.read_parquet(parquet_path)

        # Column validation — fail loudly (build.md §6)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Crypto parquet is missing required columns: {sorted(missing)}. "
                "Verify the source data matches the schema in build.md §2.2."
            )

        # Optionally restrict to a day-window (build.md §2.2)
        if window_days is not None:
            start_day, end_day = window_days
            rows_per_day = 345_600  # 250ms interval → 4 rows/s → 345,600 rows/day
            df = df.iloc[(start_day - 1) * rows_per_day: end_day * rows_per_day].copy()

        df = to_relative_price(df)
        X, y = apply_horizon_labeling(df, horizon, threshold)

        # Chronological 70/15/15 split — no shuffling across boundaries (build.md §6)
        n = len(X)
        tr  = int(n * 0.70)
        val = int(n * 0.85)

        self.X_train = X[:tr]
        self.y_train = y[:tr]
        self.X_val   = X[tr:val]
        self.y_val   = y[tr:val]
        self.X_test  = X[val:]
        self.y_test  = y[val:]

    def get_splits(self):
        """Returns (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays."""
        return (
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            self.X_test,  self.y_test,
        )

