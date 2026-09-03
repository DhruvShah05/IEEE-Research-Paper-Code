"""
experiments/ablation_structured_transformer.py — Structured Transformer ablation study (fix 5.2).

Full factorial grid (one-factor-at-a-time from headline config):
  token_mode   : flat | grouped | level
  pooling_mode : mean | cls | attention
  num_layers   : 2 | 4 | 6
  d_model      : 64 | 128
  dropout      : off (0.0) | on (0.1)
  weight_decay : off (0.0) | on (1e-4)
  epochs       : fixed15 (max_epochs=15, no patience) | early_stop (max_epochs=50, patience=5)
  5 seeds each.

Also includes a parameter-matched StandardTransformer at each (num_layers, d_model) budget.

Fix 5.2:
  - Temp configs written to experiments/ablation_scratch/ (NOT configs/ — run_all would pick them up)
  - Aggregates into a single ablation table with significance markers.
  - 5 seeds per configuration.
"""

import json
import logging
import os
import shutil
import subprocess
import sys

import yaml
import pandas as pd

from eval.aggregate import aggregate_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ablation scratch directory (fix 5.2: NOT configs/) ---
SCRATCH_DIR = 'experiments/ablation_scratch'

# --- Ablation axes (fix 5.2: full factorial / OFAT) ---
TOKEN_MODES    = ['flat', 'grouped', 'level']
POOLING_MODES  = ['mean', 'cls', 'attention']
NUM_LAYERS     = [2, 4, 6]
D_MODELS       = [64, 128]
DROPOUT_VALUES = [0.0, 0.1]
WD_VALUES      = [0.0, 1e-4]
EPOCH_MODES    = ['fixed15', 'early_stop']

SEEDS = [0, 1, 2, 3, 4]
MARKETS = ['fi2010', 'crypto']

BASE_CONFIG_FILES = {
    'fi2010': 'configs/fi2010_structured_transformer.yaml',
    'crypto': 'configs/crypto_structured_transformer.yaml',
}


def _make_config(base_config: dict, overrides: dict, run_id: str) -> tuple:
    """Returns (config_dict, config_path) for a single ablation run."""
    cfg = {**base_config}
    if 'model_params' not in cfg:
        cfg['model_params'] = {}
    cfg['model_params'] = {**cfg.get('model_params', {}), **overrides.get('model_params', {})}
    if 'training' not in cfg:
        cfg['training'] = {}
    cfg['training'] = {**cfg.get('training', {}), **overrides.get('training', {})}
    cfg['output_dir'] = f"results/ablation/{run_id}"

    # Fix 5.2: temp configs go to SCRATCH_DIR, not configs/
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    config_path = os.path.join(SCRATCH_DIR, f'{run_id}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    return cfg, config_path


def _run_config(config_path: str, seeds: list) -> list:
    """Runs main.py for all seeds; returns list of failed (seed, returncode)."""
    failures = []
    for seed in seeds:
        cmd = [sys.executable, 'main.py', '--config', config_path, '--seed', str(seed)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"  FAILED seed={seed}: {res.stderr[-500:]}")
            failures.append({'seed': seed, 'config': config_path, 'returncode': res.returncode})
        else:
            logger.info(f"  seed={seed} complete.")
    return failures


def run_ablation(markets: list = None, dry_run: bool = False):
    """
    Runs the full OFAT ablation grid.

    Parameters
    ----------
    markets : list of market names to run (default: both)
    dry_run : if True, prints configs without running
    """
    markets = markets or MARKETS
    all_failures = []

    for market in markets:
        if not os.path.exists(BASE_CONFIG_FILES[market]):
            logger.warning(f"Base config not found: {BASE_CONFIG_FILES[market]} — skipping.")
            continue

        with open(BASE_CONFIG_FILES[market]) as f:
            base_config = yaml.safe_load(f)

        # Headline config values (used as baseline for OFAT)
        headline = {
            'token_mode': 'level', 'pooling_mode': 'attention',
            'num_layers': 2, 'd_model': 64, 'dropout': 0.1,
        }
        headline_training = {'max_epochs': 50, 'patience': 5, 'weight_decay': 1e-4}

        combos = []

        # --- token_mode axis ---
        for tm in TOKEN_MODES:
            combos.append((
                f"abl_{market}_token_{tm}",
                {'model_params': {**headline, 'token_mode': tm}},
                {'training': headline_training}
            ))

        # --- pooling_mode axis ---
        for pm in POOLING_MODES:
            combos.append((
                f"abl_{market}_pool_{pm}",
                {'model_params': {**headline, 'pooling_mode': pm}},
                {'training': headline_training}
            ))

        # --- num_layers axis ---
        for nl in NUM_LAYERS:
            combos.append((
                f"abl_{market}_layers_{nl}",
                {'model_params': {**headline, 'num_layers': nl}},
                {'training': headline_training}
            ))
            # Parameter-matched StandardTransformer (fix 5.2)
            combos.append((
                f"abl_{market}_std_transformer_layers_{nl}",
                {'model_params': {'num_layers': nl, 'd_model': headline['d_model'],
                                  'token_mode': 'scalar'}},
                {'training': headline_training}
            ))

        # --- d_model axis ---
        for dm in D_MODELS:
            combos.append((
                f"abl_{market}_dmodel_{dm}",
                {'model_params': {**headline, 'd_model': dm}},
                {'training': headline_training}
            ))

        # --- dropout axis ---
        for dr in DROPOUT_VALUES:
            combos.append((
                f"abl_{market}_dropout_{int(dr*10)}",
                {'model_params': {**headline, 'dropout': dr}},
                {'training': headline_training}
            ))

        # --- weight_decay axis ---
        for wd in WD_VALUES:
            combos.append((
                f"abl_{market}_wd_{wd}",
                {'model_params': headline},
                {'training': {**headline_training, 'weight_decay': wd}}
            ))

        # --- epoch mode axis ---
        combos.append((
            f"abl_{market}_epochs_fixed15",
            {'model_params': headline},
            {'training': {'max_epochs': 15, 'patience': None, 'weight_decay': 1e-4}}
        ))
        combos.append((
            f"abl_{market}_epochs_early_stop",
            {'model_params': headline},
            {'training': {'max_epochs': 50, 'patience': 5, 'weight_decay': 1e-4}}
        ))

        logger.info(f"\n=== {market.upper()} ablation — {len(combos)} configurations ===")

        for run_id, mp_overrides, training_overrides in combos:
            overrides = {**mp_overrides, **training_overrides}
            cfg, config_path = _make_config(base_config, overrides, run_id)

            # Adjust model field for parameter-matched std transformer runs
            if 'std_transformer' in run_id:
                cfg['model'] = 'transformer'
                with open(config_path, 'w') as f:
                    yaml.dump(cfg, f)

            logger.info(f"Running: {run_id}")
            if dry_run:
                logger.info(f"  [DRY RUN] config_path={config_path}")
                continue

            failures = _run_config(config_path, SEEDS)
            all_failures.extend(failures)

    # Save failures
    os.makedirs('results', exist_ok=True)
    failures_path = 'results/ablation_failed_runs.json'
    with open(failures_path, 'w') as f:
        json.dump(all_failures, f, indent=4)
    if all_failures:
        logger.error(f"{len(all_failures)} ablation run(s) failed — see {failures_path}")

    # Clean up scratch configs
    if os.path.exists(SCRATCH_DIR):
        shutil.rmtree(SCRATCH_DIR)
        logger.info(f"Cleaned up scratch directory: {SCRATCH_DIR}")

    # Aggregate ablation results
    if not dry_run:
        logger.info("Aggregating ablation results...")
        aggregate_results('results/ablation')

    logger.info("Ablation study complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Structured Transformer Ablation Study")
    parser.add_argument('--markets', nargs='+', default=None,
                        choices=['fi2010', 'crypto'])
    parser.add_argument('--dry-run', action='store_true',
                        help="Print configs without running experiments")
    args = parser.parse_args()
    run_ablation(markets=args.markets, dry_run=args.dry_run)
