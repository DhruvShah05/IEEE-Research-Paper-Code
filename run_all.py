import os
import subprocess
import glob
import logging
import argparse
import sys
from eval.aggregate import aggregate_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all(smoke_test=False):
    """
    Loops over every config in configs/ and runs main.py for every seed (0 to 4).
    Then aggregates results.
    """
    config_files = glob.glob(os.path.join('configs', '*.yaml'))
    # Filter out base.yaml if it's there
    config_files = [f for f in config_files if not f.endswith('base.yaml')]
    
    seeds = [0, 1, 2, 3, 4]
    
    total_runs = len(config_files) * len(seeds)
    current_run = 1
    
    logger.info(f"Starting batch run of {total_runs} combinations...")
    
    for config in config_files:
        for seed in seeds:
            logger.info(f"[{current_run}/{total_runs}] Running {config} with seed {seed}")
            cmd = [sys.executable, 'main.py', '--config', config, '--seed', str(seed)]
            if smoke_test:
                cmd.append('--smoke-test')
            
            # Using subprocess to run each configuration completely independently, avoiding memory leaks
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Run failed for {config} seed {seed}:\n{res.stderr}")
            else:
                logger.info(f"Run {current_run} complete.")
            
            current_run += 1
            
    logger.info("All runs completed. Generating final tables...")
    aggregate_results()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all experiments.")
    parser.add_argument('--smoke-test', action='store_true', help="Run a quick smoke test")
    args = parser.parse_args()
    
    run_all(smoke_test=args.smoke_test)
