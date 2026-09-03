"""
scripts/tune_all.py — Run Optuna tuning ONCE per (model, market) and print the
YAML block to paste into configs/<market>_<model>.yaml   (fix 0.5 / A3).

    python scripts/tune_all.py                    # all 4 studies, 30 trials each
    python scripts/tune_all.py --models xgboost   # subset
    python scripts/tune_all.py --n-trials 50

Results are also written to results/tuning/<market>_<model>_best_params.json.
Seeded runs must NOT tune — main.py raises if n_estimators is null.
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import TrainOnlyScaler          # noqa: E402
from train.train_tree import tune_tree_model        # noqa: E402
from utils.seeding import set_seed                  # noqa: E402

MARKETS = ['fi2010', 'crypto']
MODELS = ['xgboost', 'random_forest']
TUNING_DIR = 'results/tuning'


def load_splits(config):
    if config['market'] == 'fi2010':
        from data.loaders import FI2010Dataset
        d = config.get('data', {})
        ds = FI2010Dataset(horizon_k=d.get('horizon_k', 10),
                           feature_set=d.get('feature_set', 'full144'))
    else:
        from data.loaders import CryptoDataset
        d = config.get('data', {})
        ds = CryptoDataset(
            horizon=d.get('horizon_events', 40),
            threshold=d.get('threshold', 0.0001),
            window_days=d.get('crypto_window_days'),
            price_mode=d.get('price_mode', 'fractional'),
            volume_transform=d.get('volume_transform', 'log1p'),
            labeling_scheme=d.get('labeling_scheme', 'point_to_point'),
        )
    X_tr, y_tr, X_v, y_v, _, _ = ds.get_splits()
    if config.get('data', {}).get('standardize', True):
        sc = TrainOnlyScaler()
        X_tr = sc.fit_transform(X_tr)
        X_v = sc.transform(X_v)
    return X_tr, y_tr, X_v, y_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', nargs='+', default=MARKETS)
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(TUNING_DIR, exist_ok=True)
    summary = {}

    for market in args.markets:
        for model in args.models:
            cfg_path = f'configs/{market}_{model}.yaml'
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
            config['seed'] = args.seed
            set_seed(args.seed)

            print(f'\n=== Tuning {model} on {market} ({args.n_trials} trials) ===')
            X_tr, y_tr, X_v, y_v = load_splits(config)
            best = tune_tree_model(config, X_tr, y_tr, X_v, y_v, n_trials=args.n_trials)

            out = os.path.join(TUNING_DIR, f'{market}_{model}_best_params.json')
            with open(out, 'w') as f:
                json.dump({'market': market, 'model': model, 'n_trials': args.n_trials,
                           'seed': args.seed, 'best_params': best}, f, indent=4)
            summary[cfg_path] = best

    print('\n\n===== PASTE INTO YAML (model_params block) =====')
    for cfg_path, best in summary.items():
        print(f'\n# {cfg_path}')
        print('model_params:')
        for k, v in best.items():
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
