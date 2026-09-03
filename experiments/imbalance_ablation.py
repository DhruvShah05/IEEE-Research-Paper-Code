"""
experiments/imbalance_ablation.py — Class imbalance strategy ablation (fix 5.8).

Compares class_weight vs focal_loss vs none for neural models on both markets.
Promised in build.md §7.2, absent from the paper.

Strategies: 'none' | 'class_weight' | 'focal_loss'
Models:     deeplob | transformer | structured_transformer
Markets:    fi2010 | crypto
Seeds:      5
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEEDS              = [0, 1, 2, 3, 4]
NEURAL_MODELS      = ['transformer', 'structured_transformer']
IMBALANCE_STRATEGIES = ['none', 'class_weight', 'focal_loss']
MARKETS            = ['fi2010', 'crypto']


def run_imbalance_ablation(
    out_dir: str = 'results/imbalance_ablation',
    dry_run: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    all_failures = []
    scratch_dir = 'experiments/imbalance_scratch'
    os.makedirs(scratch_dir, exist_ok=True)

    for market in MARKETS:
        for model in NEURAL_MODELS:
            base_cfg_path = f'configs/{market}_{model}.yaml'
            if not os.path.exists(base_cfg_path):
                logger.warning(f"Config not found: {base_cfg_path} — skipping.")
                continue

            with open(base_cfg_path) as f:
                base_cfg = yaml.safe_load(f)

            for strategy in IMBALANCE_STRATEGIES:
                cfg = dict(base_cfg)
                cfg['imbalance'] = {'strategy': strategy}
                if strategy == 'focal_loss':
                    cfg['imbalance']['focal_gamma'] = 2.0
                run_id = f'{market}_{model}_imb_{strategy}'
                cfg['output_dir'] = os.path.join(out_dir, run_id)

                tmp_path = os.path.join(scratch_dir, f'{run_id}.yaml')
                with open(tmp_path, 'w') as f:
                    yaml.dump(cfg, f)

                logger.info(f"Config: {run_id}")
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
                        logger.error(f"FAILED {run_id}/seed={seed}: {res.stderr[-300:]}")
                        all_failures.append({
                            'run_id': run_id, 'seed': seed,
                            'returncode': res.returncode
                        })

    if not dry_run:
        _aggregate_imbalance(out_dir)
        with open(os.path.join(out_dir, 'failed_runs.json'), 'w') as f:
            json.dump(all_failures, f, indent=4)

        if os.path.exists(scratch_dir):
            import shutil
            shutil.rmtree(scratch_dir)

    logger.info("Imbalance ablation complete.")


def _aggregate_imbalance(out_dir: str):
    records = []
    for metrics_path in glob.glob(os.path.join(out_dir, '*', 'seed_*', 'metrics.json')):
        parts = metrics_path.split(os.sep)
        run_id = parts[-3]
        seed   = parts[-2]

        # Parse run_id: {market}_{model}_imb_{strategy}
        toks = run_id.split('_imb_')
        feature_set = toks[1] if len(toks) >= 2 else 'unknown'

        manifest_path = os.path.join(os.path.dirname(metrics_path), 'run_manifest.json')
        market, model = 'unknown', 'unknown'
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                m = json.load(f)
                market = m.get('market', 'unknown')
                model  = m.get('model', 'unknown')

        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.pop('confusion_matrix', None)
        records.append({
            'market': market, 'model': model,
            'strategy': feature_set, 'seed': seed, **metrics
        })

    if not records:
        return
    df = pd.DataFrame(records)
    numeric_cols = [c for c in df.columns if c not in ['market', 'model', 'strategy', 'seed']]
    agg = df.groupby(['market', 'model', 'strategy'])[numeric_cols].agg(
        ['mean', 'std']
    ).reset_index()

    csv_path = os.path.join(out_dir, 'imbalance_ablation_results.csv')
    agg.to_csv(csv_path, index=False)
    logger.info(f"Imbalance ablation results saved to {csv_path}")
    print("\n--- Imbalance Strategy Ablation ---")
    print(agg.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Class imbalance ablation.")
    parser.add_argument('--out_dir', default='results/imbalance_ablation')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run_imbalance_ablation(out_dir=args.out_dir, dry_run=args.dry_run)
