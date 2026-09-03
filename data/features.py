import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class TrainOnlyScaler:
    """
    A scaler wrapper that ensures ``.fit()`` is only ever called once,
    guarding against accidental fitting on validation or test sets (data leakage).

    Also records and saves ``mean_`` / ``scale_`` to ``run_dir`` when
    ``save_stats()`` is called, satisfying the traceability requirement (1.1).
    """
    def __init__(self, use_zscore: bool = True):
        self.scaler = StandardScaler() if use_zscore else None
        self._is_fit = False

    def fit(self, X: np.ndarray) -> 'TrainOnlyScaler':
        if self._is_fit:
            raise RuntimeError(
                "TrainOnlyScaler.fit() called more than once! "
                "You are likely leaking test data."
            )
        if self.scaler is not None:
            self.scaler.fit(X)
        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("TrainOnlyScaler.transform() called before fit()!")
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save_stats(self, run_dir: str) -> None:
        """
        Saves ``mean_`` and ``scale_`` to ``<run_dir>/scaler_stats.npz``.
        Required for traceability (1.1 requirement).
        """
        if not self._is_fit:
            raise RuntimeError("TrainOnlyScaler.save_stats() called before fit()!")
        if self.scaler is None:
            return  # no-op when use_zscore=False
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, 'scaler_stats.npz')
        np.savez(path, mean_=self.scaler.mean_, scale_=self.scaler.scale_)


# ---------------------------------------------------------------------------
# Price representation transform
# ---------------------------------------------------------------------------

def to_relative_price(
    df: pd.DataFrame,
    mode: str = 'fractional',
    tick_size: float = None,
) -> pd.DataFrame:
    """
    Converts raw crypto price levels to a centered price representation.

    IMPORTANT: This function returns a **new** DataFrame and never mutates the
    columns that ``apply_horizon_labeling`` reads.  Always compute mid-price
    and labels from the original raw frame **before** calling this (fix 0.1).

    Parameters
    ----------
    df : pd.DataFrame
        Raw LOB snapshot frame.  Columns '2'–'21' = 10 bid levels (price, vol
        alternating); '22'–'41' = 10 ask levels.
    mode : str
        How to center/normalise bid/ask prices:

        'absolute_diff' — raw ``price − mid``  (the original, mis-named representation).
                          Documented honestly so the paper can claim "absolute $ difference".
        'fractional'    — ``(price − mid) / mid``  (fractional return from mid;
                          transferable across assets).  **Default.**
        'tick_units'    — ``(price − mid) / tick_size``  (requires ``tick_size``).

    tick_size : float, optional
        Required when ``mode='tick_units'``.

    Returns
    -------
    pd.DataFrame
        New frame with the same shape; price columns replaced by the chosen
        representation, volume columns untouched (apply optional log1p separately).
    """
    if mode not in ('absolute_diff', 'fractional', 'tick_units'):
        raise ValueError(
            f"mode must be 'absolute_diff', 'fractional', or 'tick_units', got {mode!r}"
        )
    if mode == 'tick_units' and tick_size is None:
        raise ValueError("tick_size must be provided when mode='tick_units'")

    res = df.copy()

    best_bid_col = '2'
    best_ask_col = '22'
    mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0

    # Bid columns: even index from 2 to 20 are prices
    for i in range(2, 22, 2):
        col = str(i)
        diff = df[col] - mid_price
        if mode == 'absolute_diff':
            res[col] = diff
        elif mode == 'fractional':
            res[col] = diff / mid_price
        else:  # tick_units
            res[col] = diff / tick_size

    # Ask columns: even index from 22 to 40 are prices
    for i in range(22, 42, 2):
        col = str(i)
        diff = df[col] - mid_price
        if mode == 'absolute_diff':
            res[col] = diff
        elif mode == 'fractional':
            res[col] = diff / mid_price
        else:  # tick_units
            res[col] = diff / tick_size

    return res


def apply_volume_transform(df: pd.DataFrame, mode: str = 'log1p') -> pd.DataFrame:
    """
    Applies an optional volume transform to reduce heavy-tailed volume distributions.

    Parameters
    ----------
    df : pd.DataFrame
        LOB frame (after or before price transform; volumes are odd columns).
    mode : str
        'log1p'  — ``log(1 + volume)``  (recommended; handles 0-volume levels).
        'none'   — no transform, return a copy as-is.

    Returns
    -------
    pd.DataFrame — new frame with volumes transformed.
    """
    if mode not in ('log1p', 'none'):
        raise ValueError(f"mode must be 'log1p' or 'none', got {mode!r}")

    res = df.copy()
    if mode == 'none':
        return res

    # Volume columns: odd indices in [3..21] (bid vol) and [23..41] (ask vol)
    vol_cols = [str(i) for i in range(3, 22, 2)] + [str(i) for i in range(23, 42, 2)]
    for col in vol_cols:
        if col in res.columns:
            res[col] = np.log1p(res[col])
    return res


# ---------------------------------------------------------------------------
# Derived microstructure features (1.1)
# ---------------------------------------------------------------------------

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds derived microstructure features from the raw 10-level LOB snapshot.

    Features added (required for the "why does crypto fail" analysis and for a
    fair comparison to FI-2010's engineered features — 1.1 requirement):

      spread_bps          : (ask1 − bid1) / mid × 10_000  (in basis points)
      ofi_1               : order-flow imbalance at level 1  (bid_vol1 − ask_vol1)
      ofi_5               : cumulative OFI at top 5 levels
      ofi_10              : cumulative OFI at all 10 levels
      depth_imb_1         : (bid_vol1 − ask_vol1) / (bid_vol1 + ask_vol1)
      depth_imb_5         : depth imbalance up to level 5
      depth_imb_10        : depth imbalance across all 10 levels
      micro_price         : bid1 + spread × (bid_vol1 / (bid_vol1 + ask_vol1))
      ret_5               : mid-price return over last 5 events
      ret_10              : mid-price return over last 10 events
      ret_40              : mid-price return over last 40 events

    Parameters
    ----------
    df : pd.DataFrame
        Raw LOB frame (columns '2'–'41' as per schema).

    Returns
    -------
    pd.DataFrame
        Original frame with additional feature columns appended.
    """
    res = df.copy()

    bid_price_cols = [str(i) for i in range(2, 22, 2)]   # '2','4',...,'20'
    bid_vol_cols   = [str(i) for i in range(3, 22, 2)]   # '3','5',...,'21'
    ask_price_cols = [str(i) for i in range(22, 42, 2)]  # '22','24',...,'40'
    ask_vol_cols   = [str(i) for i in range(23, 42, 2)]  # '23','25',...,'41'

    bid1 = df['2'].values.astype(float)
    ask1 = df['22'].values.astype(float)
    mid  = (bid1 + ask1) / 2.0

    bid_vol1 = df['3'].values.astype(float)
    ask_vol1 = df['23'].values.astype(float)

    # --- Spread ---
    res['spread_bps'] = (ask1 - bid1) / mid * 10_000

    # --- Order-flow imbalance (sum of bid_vol − ask_vol at top k levels) ---
    bid_vols = np.stack([df[c].values.astype(float) for c in bid_vol_cols], axis=1)
    ask_vols = np.stack([df[c].values.astype(float) for c in ask_vol_cols], axis=1)

    res['ofi_1']  = bid_vols[:, 0] - ask_vols[:, 0]
    res['ofi_5']  = (bid_vols[:, :5] - ask_vols[:, :5]).sum(axis=1)
    res['ofi_10'] = (bid_vols - ask_vols).sum(axis=1)

    # --- Depth imbalance = (bid − ask) / (bid + ask) at top k levels ---
    total_bid_1  = bid_vols[:, 0]
    total_ask_1  = ask_vols[:, 0]
    total_bid_5  = bid_vols[:, :5].sum(axis=1)
    total_ask_5  = ask_vols[:, :5].sum(axis=1)
    total_bid_10 = bid_vols.sum(axis=1)
    total_ask_10 = ask_vols.sum(axis=1)

    eps = 1e-12
    res['depth_imb_1']  = (total_bid_1  - total_ask_1)  / (total_bid_1  + total_ask_1  + eps)
    res['depth_imb_5']  = (total_bid_5  - total_ask_5)  / (total_bid_5  + total_ask_5  + eps)
    res['depth_imb_10'] = (total_bid_10 - total_ask_10) / (total_bid_10 + total_ask_10 + eps)

    # --- Micro-price: bid + spread × (bid_vol / (bid_vol + ask_vol)) ---
    spread = ask1 - bid1
    imb = bid_vol1 / (bid_vol1 + ask_vol1 + eps)
    res['micro_price'] = bid1 + spread * imb

    # --- Recent mid-price returns over 5 / 10 / 40 events ---
    mid_series = pd.Series(mid, index=df.index)
    res['ret_5']  = mid_series.pct_change(5).values
    res['ret_10'] = mid_series.pct_change(10).values
    res['ret_40'] = mid_series.pct_change(40).values

    return res
