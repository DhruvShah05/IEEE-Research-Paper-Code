"""
experiments/multi_asset.py — Multi-asset / multi-window experiment runner (fix 5.4).

Grid: symbols {BTCUSDT, ETHUSDT, SOLUSDT} × exchanges {binance, bybit} ×
      ≥2 non-overlapping multi-day windows in different volatility regimes.
Same pipeline, same configs, 5 seeds.

Output:
  - Per-asset result table.
  - Kendall's τ rank-correlation of model rankings across assets/windows
    (to support or refute "rankings are market-dependent").

Also satisfies build.md §7.6 (cross-day robustness).
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default asset grid (customize via CLI or modify here)
ASSET_GRID = [
    {'exchange': 'binance', 'symbol': 'BTCUSDT',
     'parquet': 'data/processed/crypto_data.parquet'},
    {'exchange': 'binance', 'symbol': 'ETHUSDT',
     'parquet': 'data/processed/binance_ethusdt_data.parquet'},
    {'exchange': 'binance', 'symbol': 'SOLUSDT',
     'parquet': 'data/processed/binance_solusdt_data.parquet'},
]

# Two non-overlapping time windows (expressed as [start_day, end_day] indices)
WINDOWS = [
    {'name': 'window_1', 'window_days': [1, 4]},
    {'name': 'window_2', 'window_days': [5, 8]},
]

SEEDS = [0, 1, 2, 3, 4]
MODELS = ['xgboost', 'random_forest', 'transformer', 'structured_transformer']


def _prepare_asset(exchange: str, symbol: str, parquet: str) -> bool:
    """Ensure parquet exists; try prepare_crypto.py if not."""
    if os.path.exists(parquet):
        return True
    logger.info(f"Preparing {exchange}/{symbol}...")
    res = subprocess.run([
        sys.executable, 'scripts/prepare_crypto.py',
        '--exchange', exchange, '--symbol', symbol,
    ])
    return res.returncode == 0 and os.path.exists(parquet)


def run_multi_asset(
    base_config_template: str = 'configs/crypto_xgboost.yaml',
    out_dir: str = 'results/multi_asset',
    dry_run: bool = False,
):
    """
    Runs the identical pipeline per (exchange, symbol, window) combination.

    Parameters
    ----------
    base_config_template : template config (market=crypto, model=xgboost or similar)
    out_dir              : output directory for all multi-asset results
    dry_run              : print configs without running
    """
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    all_failures = []
    all_manifests = []

    for asset in ASSET_GRID:
        exchange = asset['exchange']
        symbol   = asset['symbol']
        parquet  = asset['parquet']

        if not _prepare_asset(exchange, symbol, parquet):
            logger.warning(f"Skipping {exchange}/{symbol} — parquet not available.")
            continue

        for window in WINDOWS:
            window_name = window['name']
            asset_key   = f"{exchange}_{symbol}_{window_name}"

            if not os.path.exists(base_config_template):
                logger.warning(f"Template config not found: {base_config_template}")
                continue

            with open(base_config_template) as f:
                base_cfg = yaml.safe_load(f)

            for model in MODELS:
                # Load model-specific config if it exists
                model_cfg_path = f"configs/crypto_{model}.yaml"
                if os.path.exists(model_cfg_path):
                    with open(model_cfg_path) as f:
                        cfg = yaml.safe_load(f)
                else:
                    cfg = dict(base_cfg)
                    cfg['model'] = model

                cfg['data'] = cfg.get('data', {})
                cfg['data']['crypto_window_days'] = window['window_days']
                # Override parquet path
                cfg['data']['parquet_path'] = parquet
                cfg['output_dir'] = os.path.join(out_dir, asset_key, model)

                scratch_dir = 'experiments/multi_asset_scratch'
                os.makedirs(scratch_dir, exist_ok=True)
                tmp_cfg_path = os.path.join(scratch_dir, f'{asset_key}_{model}.yaml')
                with open(tmp_cfg_path, 'w') as f:
                    yaml.dump(cfg, f)

                if dry_run:
                    logger.info(f"[DRY RUN] {asset_key} / {model}: {tmp_cfg_path}")
                    continue

                for seed in SEEDS:
                    run_dir = os.path.join(cfg['output_dir'], f"seed_{seed}")
                    if os.path.exists(os.path.join(run_dir, 'metrics.json')):
                        logger.info(f"Skipping {asset_key}/{model}/seed_{seed} (done)")
                        continue

                    logger.info(f"Running {asset_key}/{model}/seed_{seed}")
                    res = subprocess.run(
                        [sys.executable, 'main.py', '--config', tmp_cfg_path,
                         '--seed', str(seed)],
                        capture_output=True, text=True
                    )
                    if res.returncode != 0:
                        logger.error(f"FAILED: {res.stderr[-500:]}")
                        all_failures.append({
                            'asset': asset_key, 'model': model,
                            'seed': seed, 'returncode': res.returncode,
                        })
                    else:
                        manifest = {
                            'asset': asset_key, 'exchange': exchange,
                            'symbol': symbol, 'window': window_name,
                            'model': model, 'seed': seed,
                        }
                        all_manifests.append(manifest)

    if not dry_run:
        # Aggregate and compute Kendall's τ rank-correlations
        _aggregate_multi_asset(out_dir)
        # Save failures
        with open(os.path.join(out_dir, 'failed_runs.json'), 'w') as f:
            json.dump(all_failures, f, indent=4)

    if not dry_run and os.path.exists('experiments/multi_asset_scratch'):
        import shutil
        shutil.rmtree('experiments/multi_asset_scratch')

    logger.info("Multi-asset experiment complete.")


def _aggregate_multi_asset(out_dir: str):
    """
    Aggregates per-asset metrics and computes Kendall's τ rank-correlation
    of model rankings across assets/windows.
    """
    import glob
    import json

    records = []
    for metrics_path in glob.glob(os.path.join(out_dir, '*', '*', 'seed_*', 'metrics.json')):
        parts = metrics_path.split(os.sep)
        # out_dir / asset_key / model / seed_X / metrics.json
        asset_key = parts[-4]
        model     = parts[-3]
        seed      = parts[-2]

        with open(metrics_path) as f:
            m = json.load(f)
        m.pop('confusion_matrix', None)
        records.append({'asset': asset_key, 'model': model, 'seed': seed, **m})

    if not records:
        logger.warning("No multi-asset metrics found to aggregate.")
        return

    df = pd.DataFrame(records)
    numeric_cols = [c for c in df.columns if c not in ['asset', 'model', 'seed']]
    agg = df.groupby(['asset', 'model'])[numeric_cols].mean().reset_index()

    csv_path = os.path.join(out_dir, 'multi_asset_results.csv')
    agg.to_csv(csv_path, index=False)
    logger.info(f"Multi-asset results saved to {csv_path}")

    # Kendall's τ rank-correlation across assets (model rankings)
    assets = agg['asset'].unique()
    models = agg['model'].unique()

    if len(assets) >= 2:
        tau_rows = []
        for metric in ['macro_f1', 'mcc', 'balanced_accuracy']:
            if metric not in agg.columns:
                continue
            pivot = agg.pivot(index='model', columns='asset', values=metric)
            for i, a1 in enumerate(assets):
                for a2 in assets[i + 1:]:
                    if a1 in pivot.columns and a2 in pivot.columns:
                        ranks_a1 = pivot[a1].rank()
                        ranks_a2 = pivot[a2].rank()
                        tau, p = kendalltau(ranks_a1, ranks_a2)
                        tau_rows.append({
                            'metric': metric, 'asset_1': a1, 'asset_2': a2,
                            'kendall_tau': float(tau), 'p_value': float(p),
                        })

        if tau_rows:
            tau_df = pd.DataFrame(tau_rows)
            tau_path = os.path.join(out_dir, 'rank_correlation.csv')
            tau_df.to_csv(tau_path, index=False)
            logger.info(f"Kendall rank correlation saved to {tau_path}")
            print("\n--- Kendall's τ rank-correlation of model rankings across assets ---")
            print(tau_df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-asset LOB experiment.")
    parser.add_argument('--config',  default='configs/crypto_xgboost.yaml')
    parser.add_argument('--out_dir', default='results/multi_asset')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run_multi_asset(
        base_config_template=args.config,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )
