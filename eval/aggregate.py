import os
import json
import glob
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def aggregate_results(results_dir: str = 'results'):
    """
    Reads all metrics.json files across different runs and seeds, 
    groups by (market, model), and computes mean ± std for tables.
    """
    logger.info("Aggregating results...")
    
    all_runs = []
    
    # We expect structure like results/crypto_xgboost/seed_0/metrics.json
    metrics_files = glob.glob(os.path.join(results_dir, '*', 'seed_*', 'metrics.json'))
    
    for f in metrics_files:
        path_parts = f.split(os.sep)
        seed_str = path_parts[-2] # e.g. seed_0
        run_name = path_parts[-3] # e.g. crypto_xgboost
        
        # We can extract market and model if we saved config, or parse from run_name
        # For robustness, let's also load the manifest if it exists
        manifest_file = os.path.join(os.path.dirname(f), 'run_manifest.json')
        market, model = "unknown", "unknown"
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as mf:
                manifest = json.load(mf)
                market = manifest.get('market', "unknown")
                model = manifest.get('model', "unknown")
        
        with open(f, 'r') as mf:
            metrics = json.load(mf)
            
        record = {
            'market': market,
            'model': model,
            'seed': seed_str,
            **metrics
        }
        # Drop confusion matrix for the pandas table to keep it clean
        if 'confusion_matrix' in record:
            del record['confusion_matrix']
            
        all_runs.append(record)
        
    if not all_runs:
        logger.warning("No metrics found to aggregate.")
        return
        
    df = pd.DataFrame(all_runs)
    
    # Group by market and model, compute mean and std
    numeric_cols = [c for c in df.columns if c not in ['market', 'model', 'seed']]
    
    agg_df = df.groupby(['market', 'model'])[numeric_cols].agg(['mean', 'std'])
    
    # Format cleanly as "mean ± std"
    final_table = pd.DataFrame()
    for col in numeric_cols:
        final_table[col] = agg_df[col]['mean'].map('{:.4f}'.format) + " ± " + agg_df[col]['std'].map('{:.4f}'.format)
        
    final_table = final_table.reset_index()
    
    out_file = os.path.join(results_dir, 'aggregated_metrics.csv')
    final_table.to_csv(out_file, index=False)
    logger.info(f"Aggregated metrics saved to {out_file}")
    print("\n--- Final Aggregated Results ---")
    print(final_table.to_string(index=False))

if __name__ == '__main__':
    aggregate_results()
