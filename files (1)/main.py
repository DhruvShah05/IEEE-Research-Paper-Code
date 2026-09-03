import os
import sys
import subprocess
import argparse
import yaml
import json
import logging
import time
from datetime import datetime, timezone
import numpy as np
import torch
import sklearn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from data.loaders import WindowedDataset, reorder_to_canonical

from utils.seeding import set_seed
from utils.logging import get_logger
from data.features import TrainOnlyScaler
from eval.metrics import compute_all_metrics
from data.labeling import get_class_distribution


def load_data(config: dict, logger: logging.Logger):
    """
    Loads, labels, and splits data for the configured market.

    Fix 0.1 (crypto): mid-price computed from RAW columns before calling
                       to_relative_price; passed explicitly to labeling.
    Fix 0.2 (fi2010): labels remapped from official 1/2/3 → 0/1/2 convention
                       (0=Down, 1=Stat, 2=Up).
    Fix 0.6          : timestamp-based window slicing (not row-count arithmetic).

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        ts_test, mid_test, ret_test   (timestamps, mid-prices, returns for test split)
    """
    market = config['market']
    data_cfg = config.get('data', {})

    if market == 'fi2010':
        from data.loaders import FI2010Dataset

        train_path = 'data/processed/fi2010_train.npy'
        test_path  = 'data/processed/fi2010_test.npy'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.info("FI-2010 processed files not found. Running prepare_fi2010.py...")
            res = subprocess.run([sys.executable, 'scripts/prepare_fi2010.py'])
            if res.returncode != 0:
                raise RuntimeError("Failed to process FI-2010 data.")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"FI-2010 files still not found at {train_path} / {test_path}."
                )

        horizon_k   = data_cfg.get('horizon_k', 10)
        feature_set = data_cfg.get('feature_set', 'full144')
        logger.info(
            f"FI-2010: horizon_k={horizon_k}, feature_set={feature_set} "
            "(labels remapped: 1=Up,2=Stat,3=Down → 0=Down,1=Stat,2=Up)"
        )

        ds = FI2010Dataset(
            train_path=train_path,
            test_path=test_path,
            horizon_k=horizon_k,
            feature_set=feature_set,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = ds.get_splits()

        return dict(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            ts_test=None, mid_test=None, ret_test=None,
            ds=ds,
        )

    elif market == 'crypto':
        from data.loaders import CryptoDataset

        crypto_path = 'data/processed/crypto_data.parquet'
        if not os.path.exists(crypto_path):
            logger.info("Crypto processed file not found. Running prepare_crypto.py...")
            res = subprocess.run([sys.executable, 'scripts/prepare_crypto.py'])
            if res.returncode != 0:
                raise RuntimeError("Failed to process Crypto data.")
            if not os.path.exists(crypto_path):
                raise FileNotFoundError(
                    f"Crypto file still not found at {crypto_path}."
                )

        horizon   = data_cfg.get('horizon_events', 40)
        threshold = data_cfg.get('threshold', 0.0001)
        price_mode = data_cfg.get('price_mode', 'fractional')
        vol_transform = data_cfg.get('volume_transform', 'log1p')
        labeling_scheme = data_cfg.get('labeling_scheme', 'point_to_point')

        # Fix 0.6: pass window as UNIX ms timestamps when available
        window = data_cfg.get('crypto_window_days', None)

        logger.info(
            f"Crypto: horizon={horizon}, threshold=±{threshold}, "
            f"price_mode={price_mode}, labeling={labeling_scheme}"
        )

        ds = CryptoDataset(
            parquet_path=crypto_path,
            horizon=horizon,
            threshold=threshold,
            window_days=window,
            price_mode=price_mode,
            volume_transform=vol_transform,
            labeling_scheme=labeling_scheme,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = ds.get_splits()

        return dict(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            ts_test=ds.ts_test, mid_test=ds.mid_test, ret_test=ds.ret_test,
            ds=ds,
        )

    else:
        raise ValueError(f"Unknown market: {market!r}. Must be 'fi2010' or 'crypto'.")


def main():
    parser = argparse.ArgumentParser(description="Run one (model, market, seed) experiment.")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument(
        '--smoke-test', action='store_true',
        help="Run a quick smoke test with reduced data and 1 epoch"
    )
    args = parser.parse_args()

    # 1. SET SEED FIRST — before any data loading or model construction (build.md §6)
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
    data = load_data(config, logger)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    ts_test, mid_test, ret_test = data['ts_test'], data['mid_test'], data['ret_test']
    ds = data.get('ds')  # dataset object (for save_split_manifest)

    # Fix 3.3: smoke-test operates on a COPY so original arrays are unchanged
    if args.smoke_test:
        logger.info("SMOKE TEST: Truncating dataset and reducing training iterations.")
        X_train = X_train[:100].copy()
        y_train = y_train[:100].copy()
        X_val   = X_val[:50].copy()
        y_val   = y_val[:50].copy()
        X_test  = X_test[:50].copy()
        y_test  = y_test[:50].copy()

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

        # For windowed models (DeepLOB / temporal Transformer), ensure the
        # smoke-test slice has at least T+1 rows so WindowedDataset is non-empty.
        _T = config.get('model_params', {}).get('T', 100)
        _min_rows = _T + 1
        if X_train.shape[0] < _min_rows:
            logger.warning(
                f"SMOKE TEST: X_train has only {X_train.shape[0]} rows but windowed "
                f"models need >= {_min_rows}. Padding to {_min_rows}."
            )
            # Repeat the last row to reach the minimum
            _pad = _min_rows - X_train.shape[0]
            X_train = np.vstack([X_train, np.tile(X_train[-1:], (_pad, 1))])
            y_train = np.concatenate([y_train, np.full(_pad, y_train[-1])])

    model_type = config.get('model')
    is_neural = model_type in ['deeplob', 'transformer', 'structured_transformer']

    # Determine model-specific config values needed before scaling.
    mp_cfg = config.get('model_params', {})
    _variant    = mp_cfg.get('variant', 'windowed') if model_type == 'deeplob' else None
    _token_mode = mp_cfg.get('token_mode', 'scalar') if model_type == 'transformer' else None
    is_windowed = (
        (model_type == 'deeplob' and _variant != 'snapshot')
        or (model_type == 'transformer' and _token_mode == 'temporal')
    )
    _T = mp_cfg.get('T', 100)  # sequence length for windowed models
    market = config.get('market', '')

    # A2: Determine if columns need reorder to canonical interleaved layout.
    # DeepLOB's Conv2d stride-2 kernels assume FI-2010's interleaved layout.
    # StructuredTransformer 'level'/'grouped' token modes also assume it.
    _needs_reorder = (
        (model_type == 'deeplob' and _variant != 'snapshot')
        or (model_type in ('transformer', 'structured_transformer')
            and mp_cfg.get('token_mode') in ('level', 'grouped'))
    )

    # 3. Scaling — fit ONLY on training data (build.md §8 rule 7)
    if config.get('data', {}).get('standardize', True):
        logger.info("Fitting Z-score scaler on training data only...")
        scaler = TrainOnlyScaler(use_zscore=True)
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        # Save scaler stats for traceability (1.1)
        scaler.save_stats(run_dir)

    # A2: reorder columns to canonical interleaved layout after scaling.
    if _needs_reorder:
        logger.info(f"A2: Reordering columns to canonical layout for market='{market}'.")
        X_train = reorder_to_canonical(X_train, market)
        X_val   = reorder_to_canonical(X_val,   market)
        X_test  = reorder_to_canonical(X_test,  market)

    # B4: save split manifest to run_dir so split_manifest.json is written per run.
    if ds is not None and hasattr(ds, 'save_split_manifest'):
        ds.save_split_manifest(run_dir)


    # 4. Training
    if is_neural:
        from train.train_neural import train_neural_model

        batch_size = config.get('training', {}).get('batch_size', 256)
        seed = args.seed

        # Deterministic DataLoader (fix 3.1)
        def seed_worker(worker_id):
            import random
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        # A1: Use WindowedDataset for DeepLOB (windowed) and temporal Transformer.
        # WindowedDataset yields (B, T, F); DeepLOB.forward handles the channel dim.
        # For other models keep the flat TensorDataset.
        if is_windowed:
            from data.loaders import WindowedDataset
            logger.info(f"A1: Using WindowedDataset(T={_T}) for windowed model.")
            train_ds = WindowedDataset(X_train, y_train, T=_T).dataset
            val_ds   = WindowedDataset(X_val,   y_val,   T=_T).dataset
            test_ds_obj = WindowedDataset(X_test, y_test, T=_T)
            test_ds  = test_ds_obj.dataset
            # Align ts/mid/ret/y_test with the windowed test set (first T-1 dropped).
            if ts_test is not None:
                ts_test  = ts_test[_T - 1:]
            if mid_test is not None:
                mid_test = mid_test[_T - 1:]
            if ret_test is not None:
                ret_test = ret_test[_T - 1:]
            y_test = y_test[_T - 1:]
        else:
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

        # pin_memory speeds up Host→GPU transfer on CUDA machines.
        # num_workers: 4 workers is a good default for a single GPU; avoids
        # the DataLoader becoming the bottleneck for small models.
        _pin   = torch.cuda.is_available()
        _nw    = config.get('training', {}).get('num_workers', 4 if _pin else 0)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            generator=g, worker_init_fn=seed_worker,
            pin_memory=_pin, num_workers=_nw, persistent_workers=(_nw > 0)
        )
        val_loader  = DataLoader(val_ds,  batch_size=batch_size * 4, shuffle=False,
                                 pin_memory=_pin, num_workers=_nw, persistent_workers=(_nw > 0))
        test_loader = DataLoader(test_ds, batch_size=batch_size * 4, shuffle=False,
                                 pin_memory=_pin, num_workers=_nw, persistent_workers=(_nw > 0))

        model = train_neural_model(config, train_loader, val_loader, run_dir)

        # 5. Test inference — collect both argmax and probabilities (3.1/3.3)
        model.eval()
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        model = model.to(device)
        test_preds, test_probs_list = [], []
        latency_ms_list = []

        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                t0 = time.perf_counter()
                logits = model(batch_x)
                latency_ms_list.append((time.perf_counter() - t0) * 1000 / len(batch_x))
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_probs_list.append(probs.cpu().numpy())

        test_preds = np.array(test_preds)
        test_probs = np.vstack(test_probs_list)  # shape (N_test, 3)
        avg_latency_ms = float(np.mean(latency_ms_list))

    elif model_type == 'baseline':
        # A5: chance-level / linear baselines. Same outputs as every other model
        # so they appear as rows in aggregate.py, significance and the backtest.
        from models.baselines import build_model as build_baseline_model

        model = build_baseline_model(config)
        model.fit(X_train, y_train)
        t0 = time.perf_counter()
        # Persistence needs the raw (untransformed) test mid-prices; others ignore kwargs.
        test_probs = model.predict_proba(X_test, mid_prices=mid_test)
        avg_latency_ms = (time.perf_counter() - t0) * 1000 / max(len(X_test), 1)
        test_preds = test_probs.argmax(axis=1).astype(np.int64)

        if config.get('model_params', {}).get('name') == 'persistence' and mid_test is None:
            logger.warning(
                "Persistence baseline has no mid-price series for this market — "
                "it degenerates to the majority-class baseline. Report it as such."
            )

        with open(os.path.join(run_dir, 'tuned_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    else:
        from train.train_tree import train_tree_model

        # A3: n_estimators must be frozen in the YAML (not null).
        mp = config.get('model_params', {})
        if mp.get('n_estimators') is None:
            raise ValueError(
                f"model_params.n_estimators is null in {args.config}. "
                "Run `make tune` once and paste the printed params into the YAML."
            )

        model = train_tree_model(config, X_train, y_train, X_val, y_val, run_dir)
        test_probs = model.predict_proba(X_test)  # shape (N_test, 3)
        test_preds = test_probs.argmax(axis=1).astype(np.int64)
        avg_latency_ms = None

    # 6. Metrics
    metrics = compute_all_metrics(y_test, test_preds, test_probs)
    logger.info(f"Final Test Macro-F1:  {metrics['macro_f1']:.4f}")
    logger.info(f"Final Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Final Test Bal-Acc:   {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Final Test MCC:       {metrics['mcc']:.4f}")

    # Add class distributions per split (3.3)
    metrics['train_class_dist'] = get_class_distribution(y_train)
    metrics['val_class_dist']   = get_class_distribution(y_val)
    metrics['test_class_dist']  = get_class_distribution(y_test)

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 7. Save run manifest — Fix: use timezone-aware datetime (3.3)
    try:
        param_count = int(sum(p.numel() for p in model.parameters()))
    except (AttributeError, TypeError):
        param_count = None  # tree models / baselines

    manifest = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'market': config['market'],
        # 'model' is the grouping key used by aggregate / significance / backtest.
        # Baselines get a distinct key per baseline so they are not lumped together.
        'model': (
            f"baseline_{config.get('model_params', {}).get('name')}"
            if config['model'] == 'baseline' else config['model']
        ),
        'model_config_key': config['model'],
        # Fix 0.3: avoid "Improved Transformer" — call it "Structured Transformer"
        'model_display_name': (
            'Structured Transformer' if config['model'] == 'structured_transformer'
            else f"baseline_{config.get('model_params', {}).get('name')}"
            if config['model'] == 'baseline'
            else config['model']
        ),
        'seed': args.seed,
        'config_file': args.config,
        'horizon_events': config.get('data', {}).get('horizon_events'),
        'horizon_k': config.get('data', {}).get('horizon_k'),
        'param_count': param_count,
        'avg_inference_latency_ms': avg_latency_ms,
        'library_versions': {
            'torch': torch.__version__,
            'sklearn': sklearn.__version__,
            'xgboost': xgb.__version__,
            'numpy': np.__version__,
        }
    }
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)

    # 8. Save predictions and supporting arrays (3.3 — needed for backtest)
    np.save(os.path.join(run_dir, 'test_predictions.npy'), test_preds)
    np.save(os.path.join(run_dir, 'test_labels.npy'),      y_test)
    np.save(os.path.join(run_dir, 'test_probs.npy'),       test_probs)

    if ts_test is not None:
        np.save(os.path.join(run_dir, 'test_timestamps.npy'), ts_test)
    if mid_test is not None:
        np.save(os.path.join(run_dir, 'test_mid_prices.npy'), mid_test)
    if ret_test is not None:
        np.save(os.path.join(run_dir, 'test_returns.npy'), ret_test)

    logger.info(f"Run completed. Results saved to {run_dir}")


if __name__ == '__main__':
    main()
