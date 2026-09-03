import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Convention (both markets after remapping):
#   0 = Down, 1 = Stationary, 2 = Up
#
# FI-2010 official labels are 1=Up, 2=Stationary, 3=Down (Ntakaris et al. 2018).
# Use remap_fi2010_labels() to convert to the shared convention before any
# downstream use.
# ---------------------------------------------------------------------------

FI2010_REMAP = {1: 2, 2: 1, 3: 0}   # official → shared convention


def remap_fi2010_labels(y: np.ndarray) -> np.ndarray:
    """
    Remaps FI-2010 official integer labels (1=Up, 2=Stat, 3=Down) to the
    shared convention (0=Down, 1=Stat, 2=Up).

    Parameters
    ----------
    y : np.ndarray of int
        Raw FI-2010 label array (values 1, 2, 3).

    Returns
    -------
    np.ndarray of int
        Remapped label array (values 0, 1, 2).
    """
    out = np.empty_like(y, dtype=np.int64)
    for src, dst in FI2010_REMAP.items():
        out[y == src] = dst
    return out


def label_by_threshold(returns: np.ndarray, threshold: float) -> np.ndarray:
    """
    Given a 1D array of future returns, labels them:
    0: Down   (return < -threshold)
    1: Stationary (-threshold <= return <= threshold)
    2: Up     (return > threshold)
    """
    labels = np.ones_like(returns, dtype=np.int64)  # Default to Stationary (1)
    labels[returns > threshold] = 2   # Up
    labels[returns < -threshold] = 0  # Down
    return labels


def apply_horizon_labeling(
    df: pd.DataFrame,
    horizon: int,
    threshold: float,
    mid_price: pd.Series = None,
    scheme: str = 'point_to_point',
) -> tuple:
    """
    Computes future returns for a given horizon and applies thresholding.

    Fix 0.1: accepts an explicit ``mid_price`` series computed from *raw*
    price columns.  When not provided the function falls back to deriving
    mid-price from columns '2' and '22' of ``df`` — which is only safe if
    ``df`` has NOT been transformed by ``to_relative_price`` yet.

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame (may already be relative-price transformed).
    horizon : int
        Number of events ahead for the label.
    threshold : float
        ±fractional return threshold for Up/Down classification.
    mid_price : pd.Series, optional
        Pre-computed mid-price series from **raw** columns.  Pass this when
        ``df`` has already been transformed (fix 0.1).
    scheme : str
        'point_to_point' (default): compare mid[t+H] with mid[t].
        'smoothed_mean'           : compare mean(mid[t+1..t+H]) with mid[t],
                                    matching the FI-2010 labeling convention
                                    (Ntakaris et al. 2018).

    Returns
    -------
    X : np.ndarray, shape (N-horizon, 40)
    y : np.ndarray, shape (N-horizon,)
    returns : np.ndarray, shape (N-horizon,)  — raw fractional returns
    """
    if mid_price is None:
        # Safe only if df has NOT been transformed.
        best_bid_col = '2'
        best_ask_col = '22'
        mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0
        logger.warning(
            "apply_horizon_labeling: mid_price derived from df columns '2'/'22'. "
            "Pass an explicit mid_price series if df has already been transformed "
            "(bug 0.1 — labels will be garbage otherwise)."
        )

    # Validate raw mid_price
    assert mid_price.min() > 0, (
        "mid_price contains non-positive values — check that raw (untransformed) "
        "columns '2'/'22' were used to compute mid_price."
    )

    if scheme == 'smoothed_mean':
        # Mean of next H mid-prices vs current mid (FI-2010 convention)
        future_mean = pd.Series(
            [mid_price.iloc[i + 1: i + horizon + 1].mean() for i in range(len(mid_price))],
            index=mid_price.index
        )
        returns = (future_mean - mid_price) / mid_price
    else:
        # Point-to-point: mid[t+H] vs mid[t]
        future_mid = mid_price.shift(-horizon)
        returns = (future_mid - mid_price) / mid_price

    # Drop the last `horizon` rows where future returns are NaN
    df_valid = df.iloc[:-horizon].copy()
    returns_valid = returns.iloc[:-horizon].values

    # Sanity checks (fix 0.1)
    if np.any(np.isnan(returns_valid)):
        raise ValueError(
            f"returns contain {np.sum(np.isnan(returns_valid))} NaN values after "
            "horizon labeling — mid_price likely computed from transformed columns."
        )
    if np.any(np.isinf(returns_valid)):
        raise ValueError(
            f"returns contain {np.sum(np.isinf(returns_valid))} Inf values — "
            "division by zero in mid_price calculation."
        )

    labels = label_by_threshold(returns_valid, threshold)

    # Extract just the 40 feature columns
    feature_cols = [str(i) for i in range(2, 42)]
    X = df_valid[feature_cols].values

    return X, labels, returns_valid


def apply_adaptive_threshold_labeling(
    df: pd.DataFrame,
    horizon: int,
    c: float = 1.0,
    mid_price: pd.Series = None,
    rolling_window: int = 1000,
) -> tuple:
    """
    Adaptive / volatility-scaled threshold labeling.

    Threshold = c × rolling std of returns (computed on training data).
    This keeps the class distribution roughly balanced across different
    volatility regimes (promised in §IV and future work sections).

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame (may already be relative-price transformed).
    horizon : int
        Number of events ahead for the label.
    c : float
        Multiplier on rolling std. Default 1.0.
    mid_price : pd.Series, optional
        Pre-computed mid-price series from **raw** columns.
    rolling_window : int
        Look-back window (in events) for rolling std.

    Returns
    -------
    X, y, returns, thresholds — same shapes as apply_horizon_labeling,
    plus the per-sample adaptive threshold series.
    """
    if mid_price is None:
        best_bid_col = '2'
        best_ask_col = '22'
        mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0

    future_mid = mid_price.shift(-horizon)
    returns = (future_mid - mid_price) / mid_price

    rolling_std = returns.rolling(rolling_window, min_periods=1).std()
    thresholds = c * rolling_std

    df_valid = df.iloc[:-horizon].copy()
    returns_valid = returns.iloc[:-horizon].values
    thresholds_valid = thresholds.iloc[:-horizon].values

    labels = np.ones(len(returns_valid), dtype=np.int64)  # Stationary
    labels[returns_valid > thresholds_valid] = 2   # Up
    labels[returns_valid < -thresholds_valid] = 0  # Down

    feature_cols = [str(i) for i in range(2, 42)]
    X = df_valid[feature_cols].values

    return X, labels, returns_valid, thresholds_valid


def get_class_distribution(labels: np.ndarray) -> dict:
    """
    Returns a dict with absolute counts and percentages for each class.
    Class convention: 0=Down, 1=Stationary, 2=Up.
    Saves the distribution so it can be reported in the paper (§1.2 requirement).
    """
    total = len(labels)
    counts = {int(c): int(np.sum(labels == c)) for c in [0, 1, 2]}
    pcts = {int(c): round(100 * counts[c] / total, 3) for c in [0, 1, 2]}
    return {
        'total': total,
        'counts': counts,
        'pct_down': pcts[0],
        'pct_stationary': pcts[1],
        'pct_up': pcts[2],
    }


def run_threshold_sweep(df: pd.DataFrame, horizons: list, thresholds: list,
                        mid_price: pd.Series = None) -> pd.DataFrame:
    """
    Sweeps horizon × threshold combinations and computes class balances.

    Fix 0.1: accepts an explicit ``mid_price`` computed from raw columns so
    the sweep is not run on garbage relative-price columns.
    Records class distribution of the *test split* (last 15%), not the whole set.
    """
    if mid_price is None:
        best_bid_col = '2'
        best_ask_col = '22'
        mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0
        logger.warning(
            "run_threshold_sweep: mid_price derived from df — ensure df is NOT "
            "already transformed (fix 0.1)."
        )

    results = []

    for h in horizons:
        future_mid = mid_price.shift(-h)
        returns = (future_mid - mid_price) / mid_price
        returns_valid = returns.iloc[:-h].values

        # Record distribution on test split only (last 15%)
        n = len(returns_valid)
        test_start = int(n * 0.85)
        returns_test = returns_valid[test_start:]

        for t in thresholds:
            labels_test = label_by_threshold(returns_test, t)
            dist = get_class_distribution(labels_test)

            results.append({
                'horizon_events': h,
                'threshold': t,
                'pct_down': dist['pct_down'],
                'pct_stationary': dist['pct_stationary'],
                'pct_up': dist['pct_up'],
                'total_test_samples': dist['total'],
            })

    return pd.DataFrame(results)
