"""
models/random_forest.py — Random Forest classifier wrapper.

Changes (2.4):
  - Exposes min_samples_leaf and max_features via config['model_params'].
  - Adds get_feature_importances() for mean-decrease-impurity importances.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_model(config: dict) -> RandomForestClassifier:
    """
    Builds a Scikit-Learn RandomForestClassifier based on config.

    Imbalance handling: uses class_weight='balanced' when strategy='class_weight'.
    RF does NOT use sample_weight at fit() time in the main training path
    (unlike XGBoost) — this is stated explicitly to document the difference
    and avoid confusion with the threshold_sweep script (fix 0.5 / 3.2).
    """
    mp = config.get('model_params', {})
    imbalance_strat = config.get('imbalance', {}).get('strategy', 'none')

    class_weight = 'balanced' if imbalance_strat == 'class_weight' else None

    return RandomForestClassifier(
        n_estimators    = mp.get('n_estimators',      100) or 100,
        max_depth       = mp.get('max_depth',         None),  # None = unlimited (valid for RF)
        min_samples_leaf = mp.get('min_samples_leaf', 1),
        max_features    = mp.get('max_features',      'sqrt'),
        class_weight    = class_weight,
        random_state    = config.get('seed', 42),
        n_jobs          = -1,
    )


def get_feature_importances(model: RandomForestClassifier,
                             feature_names=None) -> dict:
    """
    Returns mean-decrease-impurity (MDI) feature importances.

    Parameters
    ----------
    model         : fitted RandomForestClassifier
    feature_names : list of str, optional

    Returns
    -------
    dict with key 'mdi' mapping feature name (or index) → importance float
    """
    importances = model.feature_importances_
    if feature_names is not None:
        mdi = {str(name): float(imp) for name, imp in zip(feature_names, importances)}
    else:
        mdi = {f'f{i}': float(imp) for i, imp in enumerate(importances)}

    # Sort descending for readability
    mdi = dict(sorted(mdi.items(), key=lambda kv: kv[1], reverse=True))
    return {'mdi': mdi}
