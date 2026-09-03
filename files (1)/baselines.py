"""
models/baselines.py — Chance-level and linear baselines (fix 2.5 / A5).

Every results table must contain these rows; without them the paper cannot
claim that any model beats chance.

    majority_class       : always predicts the most frequent training class.
    random               : seeded uniform-random predictions (chance floor).
    persistence          : direction of the *realised* mid-price change over the
                           last H observations (momentum). Uses only information
                           available at time t — no labels, no future prices.
                           Crypto only (FI-2010 Z-score data has no recoverable
                           mid-price; falls back to majority class there).
    logistic_regression  : L2 multinomial logistic regression (linear upper bound).

All classes expose fit / predict / predict_proba. `predict` is always
argmax(predict_proba) so the saved argmax and probabilities agree.

Config usage (see configs/*_baseline_*.yaml):

    model: baseline
    model_params:
      name: persistence          # majority_class | random | persistence | logistic_regression
"""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

N_CLASSES = 3


def _onehot(preds: np.ndarray) -> np.ndarray:
    probs = np.zeros((len(preds), N_CLASSES), dtype=np.float32)
    probs[np.arange(len(preds)), preds] = 1.0
    return probs


class MajorityClassBaseline:
    def __init__(self, **kwargs):
        self.majority_class_ = None

    def fit(self, X, y, **kwargs):
        self.majority_class_ = int(np.argmax(np.bincount(y, minlength=N_CLASSES)))
        return self

    def predict_proba(self, X, **kwargs):
        return _onehot(np.full(len(X), self.majority_class_, dtype=np.int64))

    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs).argmax(axis=1)


class RandomBaseline:
    """Uniform random. Probabilities are drawn once per call; predict = argmax."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._last_probs = None

    def fit(self, X, y, **kwargs):
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict_proba(self, X, **kwargs):
        raw = self._rng.random((len(X), N_CLASSES)).astype(np.float32)
        self._last_probs = raw / raw.sum(axis=1, keepdims=True)
        return self._last_probs

    def predict(self, X, **kwargs):
        # Reuse the last drawn probabilities if they match this X, so that
        # predict() and predict_proba() called back-to-back agree.
        if self._last_probs is None or len(self._last_probs) != len(X):
            self.predict_proba(X)
        return self._last_probs.argmax(axis=1)


class PersistenceBaseline:
    """
    Momentum / persistence: label the direction of the realised return over
    the last H observations, using the same threshold as the target labels.

        r_t = (mid[t] - mid[t-H]) / mid[t-H]
        pred = Up   if r_t >  threshold
               Down if r_t < -threshold
               Stationary otherwise

    For t < H (no look-back available) the majority training class is used.
    Requires `mid_prices` (raw, untransformed) aligned with X at predict time.
    """

    def __init__(self, horizon: int = 40, threshold: float = 0.0001, **kwargs):
        self.horizon = int(horizon)
        self.threshold = float(threshold)
        self.fallback_class_ = 1

    def fit(self, X, y, **kwargs):
        self.fallback_class_ = int(np.argmax(np.bincount(y, minlength=N_CLASSES)))
        return self

    def predict_proba(self, X, mid_prices: np.ndarray = None, **kwargs):
        n = len(X)
        preds = np.full(n, self.fallback_class_, dtype=np.int64)
        if mid_prices is None:
            logger.warning(
                "PersistenceBaseline.predict called without mid_prices — "
                "falling back to majority class for every sample."
            )
            return _onehot(preds)

        mid = np.asarray(mid_prices, dtype=np.float64)
        if len(mid) != n:
            raise ValueError(f"mid_prices length {len(mid)} != X length {n}")
        H = self.horizon
        if n > H:
            past = mid[:-H]
            now = mid[H:]
            ret = (now - past) / past
            lab = np.ones(len(ret), dtype=np.int64)
            lab[ret > self.threshold] = 2
            lab[ret < -self.threshold] = 0
            preds[H:] = lab
        return _onehot(preds)

    def predict(self, X, mid_prices: np.ndarray = None, **kwargs):
        return self.predict_proba(X, mid_prices=mid_prices).argmax(axis=1)


class LogisticRegressionBaseline:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42,
                 class_weight=None, **kwargs):
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, solver='lbfgs', random_state=seed,
            class_weight=class_weight, n_jobs=-1,
        )

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X, **kwargs):
        return self._model.predict_proba(X)

    def predict(self, X, **kwargs):
        return self.predict_proba(X).argmax(axis=1)


BASELINE_MAP = {
    'majority_class':      MajorityClassBaseline,
    'random':              RandomBaseline,
    'persistence':         PersistenceBaseline,
    'logistic_regression': LogisticRegressionBaseline,
}

BASELINE_NAMES = tuple(BASELINE_MAP.keys())


def build_baseline(name: str, seed: int = 42, **kwargs):
    if name not in BASELINE_MAP:
        raise ValueError(f"Unknown baseline {name!r}. Options: {list(BASELINE_MAP)}")
    return BASELINE_MAP[name](seed=seed, **kwargs)


def build_model(config: dict):
    """
    Factory used by main.py.  Reads config['model_params']['name'] and wires the
    horizon / threshold / class-weight settings from the rest of the config.
    """
    mp = config.get('model_params', {})
    name = mp.get('name')
    if name is None:
        raise ValueError("model: baseline requires model_params.name")

    data_cfg = config.get('data', {})
    kwargs = {}
    if name == 'persistence':
        kwargs['horizon'] = mp.get('horizon', data_cfg.get('horizon_events', data_cfg.get('horizon_k', 40)))
        kwargs['threshold'] = mp.get('threshold', data_cfg.get('threshold', 0.0001))
    elif name == 'logistic_regression':
        kwargs['C'] = mp.get('C', 1.0)
        kwargs['max_iter'] = mp.get('max_iter', 1000)
        if config.get('imbalance', {}).get('strategy') == 'class_weight':
            kwargs['class_weight'] = 'balanced'

    model = build_baseline(name, seed=config.get('seed', 42), **kwargs)
    logger.info(f"Baseline '{name}' built with {kwargs}")
    return model
