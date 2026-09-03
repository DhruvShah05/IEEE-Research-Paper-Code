"""
train/train_tree.py — Training for XGBoost and Random Forest models.

Fix 0.5: Hyperparameter tuning is now separate from seeded training.
  - When model_params.n_estimators is None, we run Optuna tuning (≥30 trials,
    TPESampler) and persist the winning params to ``results/tuning/<model>_best_params.json``.
  - In the per-seed training path, params are expected to be frozen in the YAML.
  - The Optuna study is saved separately under ``results/tuning/``.

Fixes 3.2: saves test_probs (predict_proba) and feature importances per run.
"""

import os
import json
import logging
import optuna
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import importlib

from eval.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Maps config 'model' name to the actual Python module filename under models/.
MODEL_MODULE_MAP = {
    'xgboost': 'xgboost_model',
    'random_forest': 'random_forest',
}

# Directory for tuning study output — separate from per-seed run directories
_TUNING_DIR = 'results/tuning'


def tune_tree_model(config: dict, X_train, y_train, X_val, y_val,
                    n_trials: int = 30) -> dict:
    """
    Runs Optuna hyperparameter search (TPESampler, ≥30 trials) and saves the
    winning parameters to ``results/tuning/<model>_best_params.json``.

    Fix 0.5: this function should be called *once* (on a single fixed seed),
    and the resulting params should be frozen into the config YAML before the
    5-seed seeded training run.

    Parameters
    ----------
    config   : experiment config dict (used for model name and imbalance strategy)
    X_train, y_train, X_val, y_val : training and validation splits
    n_trials : Optuna trial budget (default ≥30 per fix 0.5)

    Returns
    -------
    dict — best hyperparameter dict from the study
    """
    model_name  = config['model']
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")

    os.makedirs(_TUNING_DIR, exist_ok=True)

    def objective(trial):
        if model_name == 'xgboost':
            # Wider search space (fix 2.4 / 0.5)
            params = {
                'n_estimators':     trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
        else:  # random_forest
            params = {
                'n_estimators':    trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
                'max_depth':       trial.suggest_int('max_depth', 3, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features':    trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }

        temp_config = dict(config)
        temp_config['model_params'] = params
        temp_model = model_module.build_model(temp_config)

        fit_kwargs = {}
        # Fix A3: For RF the constructor already has class_weight='balanced',
        # so passing sample_weight here too would double-weight and not match
        # the final training path. Only pass sample_weight for XGBoost.
        if config.get('imbalance', {}).get('strategy') == 'class_weight' and model_name == 'xgboost':
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            fit_kwargs['sample_weight'] = sample_weights

        temp_model.fit(X_train, y_train, **fit_kwargs)
        preds = temp_model.predict(X_val)
        return compute_all_metrics(y_val, preds)['macro_f1']

    logger.info(
        f"Optuna tuning for {model_name}: {n_trials} trials (TPESampler). "
        "Results saved to results/tuning/."
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),  # fixed seed for tuning (fix 0.5)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best params for {model_name}: {best_params}")

    # Persist winning params (fix 0.5 — freeze into YAML after inspecting this file)
    study_path = os.path.join(_TUNING_DIR, f'{model_name}_best_params.json')
    with open(study_path, 'w') as f:
        json.dump({'best_params': best_params, 'best_value': study.best_value}, f, indent=4)
    logger.info(
        f"Winning params saved to {study_path}. "
        "Copy n_estimators/max_depth/... into your YAML config before running seeded experiments."
    )

    return best_params


def train_tree_model(config: dict, X_train, y_train, X_val, y_val, run_dir: str):
    """
    Trains an XGBoost or Random Forest model with *fixed* hyperparameters.

    Fix 0.5: if model_params.n_estimators is None (null in YAML), we run Optuna
    tuning via ``tune_tree_model()`` to find best params, then train on those.
    In a proper resubmission workflow, call tune_tree_model() once separately
    and freeze params into the YAML — this avoids HP variance polluting the 5-seed
    variance estimate.

    Saves (3.2 requirement):
      - tuned_config.json         : resolved hyperparameters
      - feature_importances.json  : gain-based feature importances
    """
    model_name  = config['model']
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")

    model_params = config.get('model_params', {})

    # A3: params must be frozen in the YAML before seeded training.
    # Do NOT silently run tuning here — HP variance would pollute seed variance.
    # Run `make tune` once to get params, then hard-code them into the YAML.
    if model_params.get('n_estimators') is None:
        raise ValueError(
            f"model_params.n_estimators is None for model '{model_name}'. "
            "Run `make tune` once to get best params, then hard-code them into "
            "the config YAML (e.g. configs/crypto_xgboost.yaml) before running "
            "seeded experiments. See LOB_v3_review_and_run_plan.md §A3."
        )

    logger.info(f"Training final {model_name} with params: {model_params}")
    model = model_module.build_model(config)

    # Imbalance handling
    # RF uses class_weight='balanced' in constructor; XGBoost uses sample_weight at fit()
    fit_kwargs = {}
    imbalance_strat = config.get('imbalance', {}).get('strategy', 'none')
    if imbalance_strat == 'class_weight' and model_name == 'xgboost':
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        fit_kwargs['sample_weight'] = sample_weights

    # XGBoost early stopping on validation split (fix 2.4 / 3.2)
    early_stop_rounds = model_params.get('early_stopping_rounds', None)
    if model_name == 'xgboost' and early_stop_rounds:
        eval_set = [(X_val, y_val)]
        fit_kwargs['eval_set'] = eval_set
        fit_kwargs['verbose'] = False
        model.set_params(early_stopping_rounds=early_stop_rounds)

    model.fit(X_train, y_train, **fit_kwargs)

    # Persist resolved hyperparameters for traceability
    with open(os.path.join(run_dir, 'tuned_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Save feature importances (3.2 requirement)
    try:
        fi_dict = model_module.get_feature_importances(model)
        with open(os.path.join(run_dir, 'feature_importances.json'), 'w') as f:
            json.dump(fi_dict, f, indent=4)
    except (AttributeError, TypeError):
        logger.debug("feature_importances not available for this model.")

    return model
