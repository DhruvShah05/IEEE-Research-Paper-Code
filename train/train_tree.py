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
# Required because 'xgboost' is a reserved name (the xgboost library itself),
# so the file is named xgboost_model.py to avoid shadowing it.
MODEL_MODULE_MAP = {
    'xgboost': 'xgboost_model',
    'random_forest': 'random_forest',
}

def train_tree_model(config: dict, X_train, y_train, X_val, y_val, run_dir: str):
    """
    Trains an XGBoost or Random Forest model, with optional Optuna hyperparameter search.

    Optuna search space (documented per build.md §7.4):
      - n_estimators: categorical {50, 100, 200}
      - max_depth: integer in [3, 10]
    Tuning objective: Macro-F1 on the validation split.
    Trials: 5 (small fixed budget; winning params logged to tuned_config.json).
    """
    model_name = config['model']
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")

    model_params = config.get('model_params', {})

    # If parameters are null in config, we run Optuna tuning on the validation split
    if model_params.get('n_estimators') is None:
        logger.info(f"Running Optuna tuning for {model_name}...")
        logger.info("Optuna search space: n_estimators ∈ {50,100,200}, max_depth ∈ [3,10]")

        def objective(trial):
            # Optuna search space — build.md §7.4
            n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200])
            max_depth = trial.suggest_int('max_depth', 3, 10)

            temp_config = dict(config)
            temp_config['model_params'] = {'n_estimators': n_estimators, 'max_depth': max_depth}
            temp_model = model_module.build_model(temp_config)

            # XGBoost multi-class imbalance via sample_weight at fit() time
            fit_kwargs = {}
            if config.get('imbalance', {}).get('strategy') == 'class_weight' and model_name == 'xgboost':
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                fit_kwargs['sample_weight'] = sample_weights

            temp_model.fit(X_train, y_train, **fit_kwargs)
            preds = temp_model.predict(X_val)
            metrics = compute_all_metrics(y_val, preds)
            return metrics['macro_f1']

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.RandomSampler(seed=config.get('seed', 42))
        )
        study.optimize(objective, n_trials=5)

        best_params = study.best_params
        logger.info(f"Optuna best params for {model_name}: {best_params}")
        config['model_params'] = best_params

        # Persist the winning hyperparameters so they are traceable (build.md §7.4)
        with open(os.path.join(run_dir, 'tuned_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    # Build and fit the final model with the resolved hyperparameters
    logger.info(f"Training final {model_name} with params: {config.get('model_params', {})}")
    model = model_module.build_model(config)

    fit_kwargs = {}
    if config.get('imbalance', {}).get('strategy') == 'class_weight' and model_name == 'xgboost':
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        fit_kwargs['sample_weight'] = sample_weights

    model.fit(X_train, y_train, **fit_kwargs)

    # Metric evaluation on test set and result saving is handled by main.py
    return model
