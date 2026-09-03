"""
experiments/horizon_sweep.py — Horizon sweep experiment (fix 5.5).

Runs headline models on:
  - Crypto: H ∈ {10, 20, 40, 100, 250} events
  - FI-2010: k ∈ {10, 20, 50, 100} (standard horizons from the dataset)
3–5 seeds each.

Plots Macro-F1 / MCC vs horizon per market (addresses "one horizon only" limitation).
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRYPTO_HORIZONS  = [10, 20, 40, 100, 250]
FI2010_HORIZONS  = [10, 20, 50, 100]
SEEDS            = [0, 1, 2, 3, 4][:3]   # 3 seeds for speed
HEADLINE_MODELS  = ['xgboost', 'structured_transformer']


def run_horizon_sweep(
    out_dir: str = 'results/horizon_sweep',
    dry_run: bool = False,
):
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    all_failures = []

    for market, horizons, horizon_key in [
        ('crypto',  CRYPTO_HORIZONS,  'horizon_events'),
        ('fi2010',  FI2010_HORIZONS,  'horizon_k'),
    ]:
        for model in HEADLINE_MODELS:
            base_cfg_path = f'configs/{market}_{model}.yaml'
            if not os.path.exists(base_cfg_path):
                logger.warning(f"Config not found: {base_cfg_path} — skipping.")
                continue

            with open(base_cfg_path) as f:
                base_cfg = yaml.safe_load(f)

            for h in horizons:
                cfg = dict(base_cfg)
                cfg.setdefault('data', {})[horizon_key] = h
                cfg['output_dir'] = os.path.join(out_dir, market, model, f'h{h}')

                scratch_dir = 'experiments/horizon_scratch'
                os.makedirs(scratch_dir, exist_ok=True)
                tmp_path = os.path.join(scratch_dir, f'{market}_{model}_h{h}.yaml')
                with open(tmp_path, 'w') as f:
                    yaml.dump(cfg, f)

                logger.info(f"Running {market}/{model} h={h}")
                if dry_run:
                    logger.info(f"[DRY RUN] {tmp_path}")
                    continue

                for seed in SEEDS:
                    run_dir = os.path.join(cfg['output_dir'], f"seed_{seed}")
                    if os.path.exists(os.path.join(run_dir, 'metrics.json')):
                        continue
                    res = subprocess.run(
                        [sys.executable, 'main.py', '--config', tmp_path, '--seed', str(seed)],
                        capture_output=True, text=True
                    )
                    if res.returncode != 0:
                        logger.error(f"FAILED {market}/{model}/h={h}/seed={seed}: {res.stderr[-300:]}")
                        all_failures.append({
                            'market': market, 'model': model, 'horizon': h, 'seed': seed,
                        })

    if not dry_run:
        _aggregate_horizon_sweep(out_dir)
        with open(os.path.join(out_dir, 'failed_runs.json'), 'w') as f:
            json.dump(all_failures, f, indent=4)

        if os.path.exists('experiments/horizon_scratch'):
            import shutil
            shutil.rmtree('experiments/horizon_scratch')

    logger.info("Horizon sweep complete.")


def _aggregate_horizon_sweep(out_dir: str):
    """Aggregates horizon sweep metrics and saves summary."""
    records = []
    for metrics_path in glob.glob(os.path.join(out_dir, '*', '*', 'h*', 'seed_*', 'metrics.json')):
        parts = metrics_path.split(os.sep)
        market = parts[-5]
        model  = parts[-4]
        h_str  = parts[-3]   # 'h10', 'h40', etc.
        horizon = int(h_str.replace('h', ''))

        with open(metrics_path) as f:
            m = json.load(f)
        m.pop('confusion_matrix', None)
        records.append({'market': market, 'model': model, 'horizon': horizon, **m})

    if not records:
        return
    df = pd.DataFrame(records)
    numeric_cols = [c for c in df.columns if c not in ['market', 'model', 'horizon']]
    agg = df.groupby(['market', 'model', 'horizon'])[numeric_cols].agg(
        ['mean', 'std']
    ).reset_index()

    csv_path = os.path.join(out_dir, 'horizon_sweep_results.csv')
    agg.to_csv(csv_path, index=False)
    logger.info(f"Horizon sweep results saved to {csv_path}")
    print("\n--- Horizon Sweep Results ---")
    print(agg.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Horizon sweep experiment.")
    parser.add_argument('--out_dir', default='results/horizon_sweep')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run_horizon_sweep(out_dir=args.out_dir, dry_run=args.dry_run)
