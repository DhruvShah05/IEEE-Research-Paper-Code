import os
import subprocess
import yaml
import logging
from eval.aggregate import aggregate_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ablation():
    """
    Runs the structured transformer ablation study.
    token_mode ∈ {flat, grouped} × pooling_mode ∈ {mean, attention}
    """
    token_modes = ['flat', 'grouped']
    pooling_modes = ['mean', 'attention']
    markets = ['fi2010', 'crypto']
    seeds = [0] # Usually run ablation on 1 or 2 seeds to save time, unless full run is needed
    
    base_config_files = {
        'fi2010': 'configs/fi2010_structured_transformer.yaml',
        'crypto': 'configs/crypto_structured_transformer.yaml'
    }
    
    logger.info("Starting Structured Transformer Ablation Study")
    
    for market in markets:
        base_config_path = base_config_files[market]
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
            
        for t_mode in token_modes:
            for p_mode in pooling_modes:
                # Create a temporary config
                temp_config = dict(base_config)
                if 'model_params' not in temp_config:
                    temp_config['model_params'] = {}
                temp_config['model_params']['token_mode'] = t_mode
                temp_config['model_params']['pooling_mode'] = p_mode
                
                run_id = f"ablation_{market}_{t_mode}_{p_mode}"
                temp_config['output_dir'] = f"results/{run_id}"
                
                temp_config_path = f"configs/temp_{run_id}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(temp_config, f)
                    
                logger.info(f"Running Ablation: Market={market}, Token={t_mode}, Pooling={p_mode}")
                for seed in seeds:
                    cmd = ['python3', 'main.py', '--config', temp_config_path, '--seed', str(seed)]
                    subprocess.run(cmd)
                    
                # Clean up temp config
                os.remove(temp_config_path)
                
    # Aggregate specific ablation results
    logger.info("Ablation runs completed. Check results/ablation_* for metrics.")

if __name__ == '__main__':
    run_ablation()
