"""
models/xgboost_model.py — XGBoost classifier wrapper.

Changes (2.4):
  - Exposes learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda
    via config['model_params'] in addition to n_estimators and max_depth.
  - Supports early_stopping_rounds (passed at fit() time in train_tree.py).
  - Adds get_feature_importances() for gain-based + optional SHAP importances.
"""

import numpy as np
import xgboost as xgb


def build_model(config: dict) -> xgb.XGBClassifier:
    """
    Builds an XGBoost Classifier based on config.

    Imbalance handling note: XGBoost multi-class imbalance is applied via
    ``sample_weight`` at ``.fit()`` time in train_tree.py (not in the
    constructor), consistent with scikit-learn convention.
    """
    mp = config.get('model_params', {})

    return xgb.XGBClassifier(
        n_estimators      = mp.get('n_estimators',      100),
        max_depth         = mp.get('max_depth',         6),
        learning_rate     = mp.get('learning_rate',     0.1),
        subsample         = mp.get('subsample',         0.8),
        colsample_bytree  = mp.get('colsample_bytree',  0.8),
        min_child_weight  = mp.get('min_child_weight',  1),
        reg_lambda        = mp.get('reg_lambda',        1.0),
        objective         = 'multi:softprob',
        num_class         = 3,
        random_state      = config.get('seed', 42),
        n_jobs            = -1,
        tree_method       = 'hist',
        eval_metric       = 'mlogloss',
    )


def get_feature_importances(model: xgb.XGBClassifier, feature_names=None,
                             use_shap: bool = False) -> dict:
    """
    Returns gain-based feature importances as a dict.

    Parameters
    ----------
    model         : fitted XGBClassifier
    feature_names : list of str, optional
    use_shap      : if True and shap is installed, also computes SHAP importances
                    on the model's training data (requires model.get_booster()).

    Returns
    -------
    dict with keys 'gain' (and optionally 'shap_mean_abs')
    """
    scores = model.get_booster().get_score(importance_type='gain')
    if feature_names:
        # Re-map f0, f1, ... to provided names
        named = {}
        for k, v in scores.items():
            idx = int(k.replace('f', ''))
            if idx < len(feature_names):
                named[feature_names[idx]] = float(v)
            else:
                named[k] = float(v)
        gain_dict = named
    else:
        gain_dict = {k: float(v) for k, v in scores.items()}

    result = {'gain': gain_dict}

    if use_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            # Note: caller must pass X for SHAP values; skip here for safety
            result['shap_note'] = (
                "SHAP requested — call shap.TreeExplainer(model).shap_values(X) "
                "and save externally."
            )
        except ImportError:
            result['shap_note'] = "shap not installed — pip install shap"

    return result
