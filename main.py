import os
import sys
import subprocess
import argparse
import yaml
import json
import logging
import datetime
import numpy as np
import torch
import sklearn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset

from utils.seeding import set_seed
from utils.logging import get_logger
from data.features import TrainOnlyScaler, to_relative_price
from eval.metrics import compute_all_metrics

def load_data(config: dict, logger: logging.Logger):
    market = config['market']
    data_cfg = config.get('data', {})

    if market == 'fi2010':
        train_path = 'data/processed/fi2010_train.npy'
        test_path = 'data/processed/fi2010_test.npy'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.info("FI-2010 processed files not found. Running prepare_fi2010.py...")
            res = subprocess.run([sys.executable, 'scripts/prepare_fi2010.py'])
            if res.returncode != 0:
                raise RuntimeError("Failed to process FI-2010 data. Please check scripts/prepare_fi2010.py output.")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"FI-2010 processed files still not found at {train_path} / {test_path}."
                )

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        # FI-2010 has 5 label columns corresponding to horizons k ∈ {10, 20, 30, 50, 100}.
        # The horizon is explicitly configured via data.horizon_k (build.md §2.1 requirement).
        # Default k=10 matches the most common published baseline for comparability.
        horizon_k = data_cfg.get('horizon_k', 10)
        logger.info(f"FI-2010: Using configured prediction horizon k={horizon_k} events ahead")

        # Columns 0–143 are features; columns 144–148 are labels for each horizon.
        horizon_to_idx = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
        if horizon_k not in horizon_to_idx:
            raise ValueError(f"horizon_k must be one of {list(horizon_to_idx.keys())}, got {horizon_k}")
        label_idx = horizon_to_idx[horizon_k]

        X_train, y_train = train_data[:, :144], train_data[:, label_idx]
        X_test, y_test = test_data[:, :144], test_data[:, label_idx]

        # Val carved from training days only (last 20%), not from test days (build.md §6)
        split_idx = int(len(X_train) * 0.8)
        X_train_final, y_train_final = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        # FI-2010 labels are 1 (down), 2 (stationary), 3 (up) → shift to 0-indexed for CrossEntropyLoss
        y_train_final = y_train_final.astype(int) - 1
        y_val = y_val.astype(int) - 1
        y_test = y_test.astype(int) - 1

        return X_train_final, y_train_final, X_val, y_val, X_test, y_test

    elif market == 'crypto':
        import pandas as pd
        from data.labeling import apply_horizon_labeling

        crypto_path = 'data/processed/crypto_data.parquet'
        if not os.path.exists(crypto_path):
            logger.info("Crypto processed file not found. Running prepare_crypto.py...")
            res = subprocess.run([sys.executable, 'scripts/prepare_crypto.py'])
            if res.returncode != 0:
                raise RuntimeError("Failed to process Crypto data. Please check scripts/prepare_crypto.py output.")
            if not os.path.exists(crypto_path):
                raise FileNotFoundError(
                    f"Crypto processed file still not found at {crypto_path}."
                )

        df = pd.read_parquet(crypto_path)

        # Optionally restrict to a contiguous day-window (build.md §2.2)
        window = data_cfg.get('crypto_window_days', None)
        if window:
            start_day, end_day = window
            # 250ms interval → 4 rows/sec → 345,600 rows/day
            start_idx = (start_day - 1) * 345_600
            end_idx = end_day * 345_600
            df = df.iloc[start_idx:end_idx].copy()
            logger.info(f"Crypto: Using rows {start_idx}–{end_idx} ({end_day - start_day + 1} day window)")

        logger.info(f"Crypto: {len(df):,} rows loaded. Applying relative-price transform...")
        df = to_relative_price(df)

        horizon = data_cfg.get('horizon_events', 40)
        threshold = data_cfg.get('threshold', 0.0001)
        logger.info(f"Crypto: Horizon={horizon} events, Threshold=±{threshold} (±{threshold*100:.4f}%)")

        X, y = apply_horizon_labeling(df, horizon, threshold)

        # Chronological 70/15/15 split — no shuffling (build.md §6)
        n = len(X)
        tr = int(n * 0.70)
        val = int(n * 0.85)
        logger.info(f"Crypto split: train={tr:,} val={val-tr:,} test={n-val:,}")

        return X[:tr], y[:tr], X[tr:val], y[tr:val], X[val:], y[val:]

    else:
        raise ValueError(f"Unknown market: {market!r}. Must be 'fi2010' or 'crypto'.")


def main():
    parser = argparse.ArgumentParser(description="Run one (model, market, seed) experiment.")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--smoke-test', action='store_true', help="Run a quick smoke test with reduced data and 1 epoch")
    args = parser.parse_args()

    # 1. SET SEED FIRST — before any data loading or model construction (build.md §6, §8 rule 6)
    set_seed(args.seed)

    logger = get_logger(__name__)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['seed'] = args.seed

    # Setup per-run output directory
    base_out_dir = config.get('output_dir', 'results/default/')
    run_dir = os.path.join(base_out_dir, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"Starting run: config={args.config}, seed={args.seed}")
    logger.info(f"Output dir: {run_dir}")

    # 2. Data Loading & Splitting
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config, logger)

    if args.smoke_test:
        logger.info("SMOKE TEST: Truncating dataset and reducing training iterations.")
        X_train, y_train = X_train[:100], y_train[:100]
        X_val, y_val = X_val[:50], y_val[:50]
        X_test, y_test = X_test[:50], y_test[:50]
        
        # Ensure all 3 classes are present so classifiers infer num_classes=3 correctly
        if len(y_train) >= 3:
            y_train[0] = 0
            y_train[1] = 1
            y_train[2] = 2
        
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = 1
        
        if 'model_params' not in config:
            config['model_params'] = {}
        config['model_params']['n_estimators'] = 2
        config['model_params']['max_depth'] = 3

    # 3. Scaling — fit ONLY on training data (build.md §8 rule 7)
    if config.get('data', {}).get('standardize', True):
        logger.info("Fitting Z-score scaler on training data only...")
        scaler = TrainOnlyScaler(use_zscore=True)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model_type = config.get('model')
    is_neural = model_type in ['deeplob', 'transformer', 'structured_transformer']

    # 4. Training
    if is_neural:
        from train.train_neural import train_neural_model

        batch_size = config.get('training', {}).get('batch_size', 256)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = train_neural_model(config, train_loader, val_loader, run_dir)

        # 5. Test inference
        model.eval()
        # Device priority: CUDA (cloud GPU) > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        test_preds = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
        test_preds = np.array(test_preds)

    else:
        from train.train_tree import train_tree_model
        model = train_tree_model(config, X_train, y_train, X_val, y_val, run_dir)
        test_preds = model.predict(X_test)

    # 6. Metrics
    metrics = compute_all_metrics(y_test, test_preds)
    logger.info(f"Final Test Macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Final Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Final Test MCC:       {metrics['mcc']:.4f}")

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 7. Save run manifest (build.md §6): timestamp, seed, library versions, key config fields
    manifest = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'market': config['market'],
        'model': config['model'],
        'seed': args.seed,
        'config_file': args.config,
        'horizon_events': config.get('data', {}).get('horizon_events'),
        'horizon_k': config.get('data', {}).get('horizon_k'),
        'library_versions': {
            'torch': torch.__version__,
            'sklearn': sklearn.__version__,
            'xgboost': xgb.__version__,
            'numpy': np.__version__,
        }
    }
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)

    # 8. Save raw predictions for significance testing (build.md §7.5)
    np.save(os.path.join(run_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(run_dir, 'test_labels.npy'), y_test)

    logger.info(f"Run completed. Results saved to {run_dir}")


if __name__ == '__main__':
    main()

