"""
experiments/threshold_sweep.py — Crypto labeling sensitivity study (fix 5.3).

Fixes:
  0.1 : Labels now computed from raw prices (mid_price derived BEFORE to_relative_price).
  5.3 : Extended grid includes smoothed labeling and adaptive thresholds.
        Class distribution recorded on the test split, not the whole set.
        Output goes to results/sweeps/ (not .gitignored).

Sweeps horizon × threshold combinations and records:
  1. Class balance (% Down / Stationary / Up) for every (horizon, threshold) pair
     — on the test split only (last 15% of data).
  2. Downstream Macro-F1 for tree models across the grid using raw-price labels.

Outputs (saved to --out_dir, default results/sweeps/):
  threshold_sweep_balance.csv  — class distribution per combination (test split)
  threshold_sweep_f1.csv       — tree-model Macro-F1 per combination
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from data.labeling import (
    run_threshold_sweep,
    apply_horizon_labeling,
    apply_adaptive_threshold_labeling,
    get_class_distribution,
)
from data.features import to_relative_price, apply_volume_transform, TrainOnlyScaler
from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_model_f1_sweep(
    df_raw: pd.DataFrame,
    horizons: list,
    thresholds: list,
    seed: int = 0,
    labeling_scheme: str = 'point_to_point',
) -> pd.DataFrame:
    """
    For every (horizon, threshold) combination, trains RF and XGBoost on a
    70/15/15 split and records test split Macro-F1.

    Fix 0.1: Uses raw df for mid_price; transforms features only AFTER labels.
    Fix 5.3: Accepts labeling_scheme for smoothed vs point-to-point comparison.
    """
    records = []

    # Compute raw mid_price ONCE from raw columns (fix 0.1)
    raw_mid_price = (df_raw['2'].astype(float) + df_raw['22'].astype(float)) / 2.0

    # Transform features (separate from labeling)
    df_transformed = to_relative_price(df_raw, mode='fractional')
    df_transformed = apply_volume_transform(df_transformed, mode='log1p')

    for h in horizons:
        for t in thresholds:
            logger.info(f"  Training models: horizon={h}, threshold={t}, scheme={labeling_scheme}")

            # Labels from raw mid_price (fix 0.1)
            X, y, returns = apply_horizon_labeling(
                df_transformed, h, t,
                mid_price=raw_mid_price, scheme=labeling_scheme
            )

            n = len(X)
            tr_end  = int(n * 0.70)
            val_end = int(n * 0.85)

            X_train, y_train = X[:tr_end],          y[:tr_end]
            X_val,   y_val   = X[tr_end:val_end],   y[tr_end:val_end]
            X_test,  y_test  = X[val_end:],         y[val_end:]

            scaler  = TrainOnlyScaler(use_zscore=True)
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)

            row = {
                'horizon_events': h,
                'threshold': t,
                'labeling_scheme': labeling_scheme,
            }

            # Class distribution on test split (fix 5.3)
            dist = get_class_distribution(y_test)
            row['test_pct_down']       = dist['pct_down']
            row['test_pct_stationary'] = dist['pct_stationary']
            row['test_pct_up']         = dist['pct_up']
            row['test_n_samples']      = dist['total']

            # --- Random Forest ---
            sw_rf = compute_sample_weight(class_weight='balanced', y=y_train)
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train, sample_weight=sw_rf)
            rf_f1 = compute_all_metrics(y_test, rf.predict(X_test))['macro_f1']
            row['rf_test_macro_f1'] = rf_f1

            # --- XGBoost ---
            sw_xgb = compute_sample_weight(class_weight='balanced', y=y_train)
            clf_xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, objective='multi:softprob', num_class=3,
                random_state=seed, n_jobs=-1, tree_method='hist', verbosity=0
            )
            clf_xgb.fit(X_train, y_train, sample_weight=sw_xgb)
            xgb_f1 = compute_all_metrics(y_test, clf_xgb.predict(X_test))['macro_f1']
            row['xgb_test_macro_f1'] = xgb_f1

            logger.info(
                f"    RF={rf_f1:.4f}  XGB={xgb_f1:.4f}  "
                f"test dist: ↓{dist['pct_down']:.1f}% —{dist['pct_stationary']:.1f}% ↑{dist['pct_up']:.1f}%"
            )
            records.append(row)

    return pd.DataFrame(records)


def run_adaptive_threshold_sweep(df_raw: pd.DataFrame, horizons: list,
                                  cs: list, seed: int = 0) -> pd.DataFrame:
    """
    Adaptive/volatility-scaled threshold sweep.
    threshold = c × rolling_std(returns), so class balance adapts to volatility.
    """
    raw_mid_price = (df_raw['2'].astype(float) + df_raw['22'].astype(float)) / 2.0
    df_transformed = to_relative_price(df_raw, mode='fractional')
    records = []

    for h in horizons:
        for c in cs:
            logger.info(f"  Adaptive threshold: horizon={h}, c={c}")
            X, y, returns, thresholds = apply_adaptive_threshold_labeling(
                df_transformed, h, c=c, mid_price=raw_mid_price
            )
            n = len(X)
            test_start = int(n * 0.85)
            dist = get_class_distribution(y[test_start:])

            row = {
                'horizon_events': h,
                'c_multiplier': c,
                'mean_threshold': float(thresholds[test_start:].mean()),
                'test_pct_down': dist['pct_down'],
                'test_pct_stationary': dist['pct_stationary'],
                'test_pct_up': dist['pct_up'],
                'test_n_samples': dist['total'],
            }
            records.append(row)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Crypto labeling threshold sensitivity sweep.")
    parser.add_argument('--input',   default='data/processed/crypto_data.parquet')
    parser.add_argument('--out_dir', default='results/sweeps',  # fix 5.3 — not gitignored
                        help="Output directory (not inside results/ root which was gitignored)")
    parser.add_argument('--seed',    type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Cannot find prepared data at {args.input}. Run prepare_crypto.py first.")
        return

    # Fix 0.1: load raw df — DO NOT call to_relative_price before this point
    df_raw = pd.read_parquet(args.input)

    # Fix 0.1: mid_price from raw columns
    raw_mid_price = (df_raw['2'].astype(float) + df_raw['22'].astype(float)) / 2.0

    # Sweep configuration
    horizons   = [10, 40, 100, 250, 500]
    thresholds = [0.0001, 0.0002, 0.0005, 0.001]
    adaptive_cs = [0.5, 1.0, 1.5, 2.0]

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Part 1: Class balance (fix 0.1: pass raw mid_price; fix 5.3: test split only) ---
    logger.info(f"Running class-balance sweep on {len(df_raw):,} rows...")
    # run_threshold_sweep internally accepts mid_price to avoid fix-0.1 bug
    balance_df = run_threshold_sweep(df_raw, horizons, thresholds, mid_price=raw_mid_price)
    balance_out = os.path.join(args.out_dir, 'threshold_sweep_balance.csv')
    balance_df.to_csv(balance_out, index=False)
    logger.info(f"Class-balance sweep saved to {balance_out}")
    print("\n--- Class Balance (test split) ---")
    print(balance_df.to_string(index=False))

    # --- Part 2: Tree model Macro-F1 (point-to-point) ---
    logger.info("\nRunning tree-model F1 sweep (point-to-point)...")
    f1_df = run_model_f1_sweep(df_raw, horizons, thresholds, seed=args.seed,
                                labeling_scheme='point_to_point')
    f1_out = os.path.join(args.out_dir, 'threshold_sweep_f1.csv')
    f1_df.to_csv(f1_out, index=False)
    logger.info(f"F1 sweep saved to {f1_out}")
    print("\n--- Tree Model Test Macro-F1 ---")
    print(f1_df.to_string(index=False))

    # --- Part 3: Smoothed labeling (fix 5.3 — labeling confound comparison) ---
    logger.info("\nRunning tree-model F1 sweep (smoothed mean labeling)...")
    f1_df_sm = run_model_f1_sweep(df_raw, horizons, thresholds, seed=args.seed,
                                   labeling_scheme='smoothed_mean')
    f1_sm_out = os.path.join(args.out_dir, 'threshold_sweep_f1_smoothed.csv')
    f1_df_sm.to_csv(f1_sm_out, index=False)
    logger.info(f"Smoothed F1 sweep saved to {f1_sm_out}")

    # --- Part 4: Adaptive threshold sweep (fix 5.3) ---
    logger.info("\nRunning adaptive threshold sweep...")
    adapt_df = run_adaptive_threshold_sweep(df_raw, horizons[:3], adaptive_cs, seed=args.seed)
    adapt_out = os.path.join(args.out_dir, 'threshold_sweep_adaptive.csv')
    adapt_df.to_csv(adapt_out, index=False)
    logger.info(f"Adaptive sweep saved to {adapt_out}")
    print("\n--- Adaptive Threshold Sweep ---")
    print(adapt_df.to_string(index=False))

    print(
        f"\nSweep complete. Results saved to {args.out_dir}. "
        "Use these results to justify the final horizon/threshold in configs/crypto_*.yaml"
    )


if __name__ == '__main__':
    main()
