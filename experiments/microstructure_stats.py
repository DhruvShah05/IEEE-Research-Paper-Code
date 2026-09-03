"""
experiments/microstructure_stats.py — LOB microstructure statistics table (fix 5.6).

For each dataset/window computes:
  - Mean/median spread in ticks and bps
  - Depth at levels 1/5/10
  - Snapshot update rate (rows/second)
  - Mid-price return volatility at the label horizon
  - Autocorrelation of mid-price returns and of OFI
  - Fraction of snapshots where |return| < threshold

Produces one table comparing FI-2010 vs each crypto asset/window.
This is the evidence for *why* crypto is harder, turning the negative result
into a contribution (5.6 requirement).
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_crypto_stats(parquet_path: str, horizon: int = 40,
                          threshold: float = 0.0001,
                          name: str = 'crypto') -> dict:
    """
    Computes microstructure statistics from a raw crypto parquet file.
    """
    df = pd.read_parquet(parquet_path)

    bid1 = df['2'].astype(float)
    ask1 = df['22'].astype(float)
    mid  = (bid1 + ask1) / 2.0

    spread_abs = ask1 - bid1
    spread_bps = spread_abs / mid * 10_000

    # Depth (sum of bid + ask volume) at levels 1, 5, 10
    def depth_at(k):
        bid_vol_cols = [str(i) for i in range(3, 3 + 2 * k, 2)]
        ask_vol_cols = [str(i) for i in range(23, 23 + 2 * k, 2)]
        bid_depth = df[bid_vol_cols].astype(float).sum(axis=1)
        ask_depth = df[ask_vol_cols].astype(float).sum(axis=1)
        return bid_depth + ask_depth

    # Snapshot update rate
    ts = df['0'].astype(float)
    median_interval_ms = float(ts.diff().dropna().median())
    update_rate_hz = 1000 / median_interval_ms if median_interval_ms > 0 else float('nan')

    # Returns at horizon
    future_mid = mid.shift(-horizon)
    returns = ((future_mid - mid) / mid).iloc[:-horizon].values

    # Autocorrelation of mid-price returns (lag 1)
    ret1 = mid.pct_change().dropna().values
    autocorr_ret = float(pd.Series(ret1).autocorr(lag=1))

    # OFI (level 1 only)
    bid_vol1 = df['3'].astype(float)
    ask_vol1 = df['23'].astype(float)
    ofi = (bid_vol1 - ask_vol1).values
    autocorr_ofi = float(pd.Series(ofi).autocorr(lag=1))

    # Fraction stationary
    frac_stat = float(np.mean(np.abs(returns) < threshold))

    return {
        'name': name,
        'n_rows': int(len(df)),
        'update_rate_hz': round(update_rate_hz, 2),
        'median_interval_ms': round(median_interval_ms, 2),
        'spread_mean_bps': round(float(spread_bps.mean()), 4),
        'spread_median_bps': round(float(spread_bps.median()), 4),
        'depth_l1_mean': round(float(depth_at(1).mean()), 2),
        'depth_l5_mean': round(float(depth_at(5).mean()), 2),
        'depth_l10_mean': round(float(depth_at(10).mean()), 2),
        'return_vol_at_horizon': round(float(np.std(returns, ddof=1)), 6),
        'autocorr_midret_lag1': round(autocorr_ret, 4),
        'autocorr_ofi_lag1': round(autocorr_ofi, 4),
        f'frac_stationary_at_thr_{threshold}': round(frac_stat, 4),
    }


def compute_fi2010_stats(npy_path: str, horizon_col_idx: int = 144,
                          name: str = 'fi2010') -> dict:
    """
    Computes microstructure statistics from a FI-2010 .npy file.

    Note: Z-score data cannot recover mid-price in absolute terms.
    We report relative statistics (e.g., return volatility at horizon).
    For spread/depth in bps, the DecPre variant is required.
    """
    data = np.load(npy_path)
    # Features 0–39 = raw LOB (DecPre only; Z-score is normalized)
    # Labels at column horizon_col_idx
    labels = data[:, horizon_col_idx].astype(int)

    # Return proxy: change in (best_bid + best_ask) / 2 — only meaningful for DecPre
    # For Z-score we report what we can
    n_rows = data.shape[0]
    n_features = data.shape[1]

    # Label-derived statistics (available for both variants)
    from data.labeling import remap_fi2010_labels
    y = remap_fi2010_labels(labels)
    n_down = int(np.sum(y == 0))
    n_stat = int(np.sum(y == 1))
    n_up   = int(np.sum(y == 2))

    return {
        'name': name,
        'n_rows': int(n_rows),
        'n_features': int(n_features),
        'frac_down': round(n_down / n_rows, 4),
        'frac_stationary': round(n_stat / n_rows, 4),
        'frac_up': round(n_up / n_rows, 4),
        'note': 'Z-score normalized; spread/depth/mid-price stats require DecPre variant',
    }


def main():
    parser = argparse.ArgumentParser(description="LOB microstructure statistics.")
    parser.add_argument('--crypto_parquet', default='data/processed/crypto_data.parquet')
    parser.add_argument('--fi2010_train',   default='data/processed/fi2010_train.npy')
    parser.add_argument('--horizon',         type=int, default=40)
    parser.add_argument('--threshold',       type=float, default=0.0001)
    parser.add_argument('--out_dir',         default='results/microstructure')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    if os.path.exists(args.crypto_parquet):
        stats = compute_crypto_stats(
            args.crypto_parquet, args.horizon, args.threshold,
            name='crypto_BTCUSDT_binance'
        )
        rows.append(stats)
        logger.info("Crypto stats computed.")
    else:
        logger.warning(f"Crypto parquet not found: {args.crypto_parquet}")

    if os.path.exists(args.fi2010_train):
        stats = compute_fi2010_stats(args.fi2010_train, name='fi2010_train')
        rows.append(stats)
        logger.info("FI-2010 stats computed.")
    else:
        logger.warning(f"FI-2010 npy not found: {args.fi2010_train}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.out_dir, 'microstructure_stats.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Microstructure stats saved to {csv_path}")
        with open(os.path.join(args.out_dir, 'microstructure_stats.json'), 'w') as f:
            json.dump(rows, f, indent=4)
        print("\n--- LOB Microstructure Statistics ---")
        print(df.T.to_string())
    else:
        logger.warning("No datasets found — run prepare scripts first.")


if __name__ == '__main__':
    main()
