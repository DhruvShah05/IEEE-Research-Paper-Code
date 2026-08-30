"""
experiments/threshold_sweep.py — Crypto labeling sensitivity study (build.md §7.1).

Sweeps horizon × threshold combinations and records:
  1. Class balance (% Down / Stationary / Up) for every (horizon, threshold) pair.
  2. Downstream Macro-F1 for both tree models (RF and XGBoost) across the grid
     using a quick train/val/test run so the final labeling choice is justified
     by performance evidence, not asserted (build.md §7.1).

Outputs (saved to --out_dir):
  threshold_sweep_balance.csv  — class distribution per combination
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

from data.labeling import run_threshold_sweep, apply_horizon_labeling
from data.features import to_relative_price, TrainOnlyScaler
from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_model_f1_sweep(df: pd.DataFrame, horizons: list, thresholds: list, seed: int = 0) -> pd.DataFrame:
    """
    For every (horizon, threshold) combination, trains a quick Random Forest and XGBoost
    on a 70% train / 15% val+test split and records validation Macro-F1.

    Trees are used because they are cheap enough to sweep the full grid (build.md §7.1).
    A single seed is used here for speed; the main experiment uses 5 seeds.
    """
    records = []

    for h in horizons:
        for t in thresholds:
            logger.info(f"  Training models: horizon={h}, threshold={t}")

            X, y = apply_horizon_labeling(df, h, t)

            n = len(X)
            tr_end  = int(n * 0.70)
            val_end = int(n * 0.85)

            X_train, y_train = X[:tr_end],       y[:tr_end]
            X_val,   y_val   = X[tr_end:val_end], y[tr_end:val_end]

            # Standardize — fit only on training split (build.md §8 rule 7)
            scaler = TrainOnlyScaler(use_zscore=True)
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            row = {'horizon_events': h, 'threshold': t}

            # --- Random Forest ---
            sw_rf = compute_sample_weight(class_weight='balanced', y=y_train)
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train, sample_weight=sw_rf)
            rf_f1 = compute_all_metrics(y_val, rf.predict(X_val))['macro_f1']
            row['rf_val_macro_f1'] = rf_f1

            # --- XGBoost ---
            sw_xgb = compute_sample_weight(class_weight='balanced', y=y_train)
            clf_xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, objective='multi:softprob', num_class=3,
                random_state=seed, n_jobs=-1, tree_method='hist', verbosity=0
            )
            clf_xgb.fit(X_train, y_train, sample_weight=sw_xgb)
            xgb_f1 = compute_all_metrics(y_val, clf_xgb.predict(X_val))['macro_f1']
            row['xgb_val_macro_f1'] = xgb_f1

            logger.info(f"    RF Macro-F1={rf_f1:.4f}  XGB Macro-F1={xgb_f1:.4f}")
            records.append(row)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Crypto labeling threshold sensitivity sweep.")
    parser.add_argument('--input',   type=str, default='data/processed/crypto_data.parquet',
                        help="Prepared crypto parquet (from prepare_crypto.py)")
    parser.add_argument('--out_dir', type=str, default='results',
                        help="Directory to save sweep CSVs")
    parser.add_argument('--seed',    type=int, default=0,
                        help="Random seed for tree model training")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Cannot find prepared data at {args.input}. Run scripts/prepare_crypto.py first.")
        return

    df = pd.read_parquet(args.input)
    df = to_relative_price(df)

    # Sweep configuration — build.md §7.1
    horizons   = [10, 40, 100, 250, 500]
    thresholds = [0.0001, 0.0002, 0.0005, 0.001]

    logger.info(f"Running class-balance sweep on {len(df):,} rows...")
    logger.info(f"Horizons: {horizons}, Thresholds: {thresholds}")

    # --- Part 1: Class balance ---
    balance_df = run_threshold_sweep(df, horizons, thresholds)
    os.makedirs(args.out_dir, exist_ok=True)
    balance_out = os.path.join(args.out_dir, 'threshold_sweep_balance.csv')
    balance_df.to_csv(balance_out, index=False)
    logger.info(f"Class-balance sweep saved to {balance_out}")
    print("\n--- Class Balance ---")
    print(balance_df.to_string(index=False))

    # --- Part 2: Tree model Macro-F1 (build.md §7.1 requirement) ---
    logger.info("\nRunning tree-model F1 sweep (this may take several minutes)...")
    f1_df = run_model_f1_sweep(df, horizons, thresholds, seed=args.seed)
    f1_out = os.path.join(args.out_dir, 'threshold_sweep_f1.csv')
    f1_df.to_csv(f1_out, index=False)
    logger.info(f"Model F1 sweep saved to {f1_out}")
    print("\n--- Tree Model Macro-F1 ---")
    print(f1_df.to_string(index=False))

    print(f"\nSweep complete. Use these results to justify the final horizon/threshold in configs/crypto_*.yaml")


if __name__ == '__main__':
    main()

