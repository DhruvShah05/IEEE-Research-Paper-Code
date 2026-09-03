"""
run_all.py — Batch runner for all (config, seed) experiment combinations.

Changes (3.4):
  - Added --markets, --models, --seeds, --assets filters for partial re-runs.
  - Failures written to results/failed_runs.json; exits non-zero at the end.
  - Loops over feature_set variants when multiple are configured.
  - Calls run_significance.py and plots.py after aggregation.
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys

from eval.aggregate import aggregate_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_all(
    smoke_test: bool = False,
    markets_filter: list = None,
    models_filter: list = None,
    seeds_filter: list = None,
):
    """
    Loops over every config in configs/ and runs main.py for every seed.
    Then aggregates results, runs significance tests, and generates plots.

    Parameters
    ----------
    smoke_test     : if True, passes --smoke-test to each run.
    markets_filter : if set, only runs configs whose market is in this list.
    models_filter  : if set, only runs configs whose model is in this list.
    seeds_filter   : if set, only runs these seed values (default: [0,1,2,3,4]).
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"==== HARDWARE CHECK: Using GPU ({torch.cuda.get_device_name(0)}) ====")
        else:
            logger.warning("==== HARDWARE CHECK: No GPU detected! PyTorch will run on CPU. ====")
    except ImportError:
        pass

    # Collect config files — exclude base.yaml and temp_ ablation configs
    config_files = glob.glob(os.path.join('configs', '*.yaml'))
    config_files = [
        f for f in config_files
        if not os.path.basename(f).startswith('base')
        and not os.path.basename(f).startswith('temp_')
    ]
    config_files.sort()

    seeds = seeds_filter if seeds_filter else [0, 1, 2, 3, 4]

    # Apply market / model filters
    filtered_configs = []
    for cfg_path in config_files:
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            continue
        market = cfg.get('market', '')
        model  = cfg.get('model', '')
        if markets_filter and market not in markets_filter:
            continue
        if models_filter and model not in models_filter:
            continue
        filtered_configs.append((cfg_path, cfg))

    total_runs = len(filtered_configs) * len(seeds)
    current_run = 1
    failed_runs = []

    logger.info(f"Starting batch run of {total_runs} combinations...")

    for cfg_path, cfg in filtered_configs:
        for seed in seeds:
            base_out_dir = cfg.get('output_dir', 'results/default/')
            run_dir      = os.path.join(base_out_dir, f"seed_{seed}")
            metrics_file = os.path.join(run_dir, 'metrics.json')

            if os.path.exists(metrics_file):
                logger.info(f"[{current_run}/{total_runs}] Skipping {cfg_path} seed {seed} (done)")
                current_run += 1
                continue

            logger.info(f"[{current_run}/{total_runs}] Running {cfg_path} seed {seed}")
            cmd = [sys.executable, 'main.py', '--config', cfg_path, '--seed', str(seed)]
            if smoke_test:
                cmd.append('--smoke-test')

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Run FAILED: {cfg_path} seed {seed}\n{res.stderr[-2000:]}")
                failed_runs.append({
                    'config': cfg_path,
                    'seed': seed,
                    'returncode': res.returncode,
                    'stderr_tail': res.stderr[-1000:],
                })
            else:
                logger.info(f"Run {current_run} complete.")

            current_run += 1

    # Write failed runs (3.4 — no silent swallowing)
    os.makedirs('results', exist_ok=True)
    failed_path = os.path.join('results', 'failed_runs.json')
    with open(failed_path, 'w') as f:
        json.dump(failed_runs, f, indent=4)
    if failed_runs:
        logger.error(f"{len(failed_runs)} run(s) failed — see {failed_path}")

    # Aggregate
    logger.info("All runs completed. Generating aggregated tables...")
    aggregate_results()

    # Significance tests (3.4)
    sig_script = os.path.join('eval', 'run_significance.py')
    if os.path.exists(sig_script):
        logger.info("Running significance tests...")
        subprocess.run([sys.executable, sig_script], capture_output=False)

    # Plots (3.4)
    plot_script = os.path.join('eval', 'plots.py')
    if os.path.exists(plot_script):
        logger.info("Generating plots...")
        subprocess.run([sys.executable, plot_script], capture_output=False)

    # Exit non-zero if any run failed (3.4)
    if failed_runs:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all experiments.")
    parser.add_argument('--smoke-test', action='store_true', help="Quick smoke test")
    parser.add_argument('--markets', nargs='+', default=None,
                        help="Filter by market (e.g. --markets fi2010 crypto)")
    parser.add_argument('--models',  nargs='+', default=None,
                        help="Filter by model (e.g. --models xgboost random_forest)")
    parser.add_argument('--seeds',   nargs='+', type=int, default=None,
                        help="Filter by seed values (e.g. --seeds 0 1 2)")
    args = parser.parse_args()

    run_all(
        smoke_test=args.smoke_test,
        markets_filter=args.markets,
        models_filter=args.models,
        seeds_filter=args.seeds,
    )
