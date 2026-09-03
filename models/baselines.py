"""
models/baselines.py — Baseline classifiers required in every result table.

Without these baselines the paper cannot claim any model beats chance (fix 2.5).
All baselines implement a scikit-learn-compatible interface (fit / predict /
predict_proba) so they slot into the existing training and evaluation pipeline.

Baselines
---------
MajorityClassBaseline   : always predicts the most frequent training class.
PersistenceBaseline     : predicts the last observed label (momentum-free baseline).
LogisticRegressionBaseline : L2-regularised logistic regression (linear model upper bound).
RandomBaseline          : seeded uniform random predictions (chance level reference).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state


class MajorityClassBaseline:
    """
    Always predicts the majority class from the training set.
    Balanced accuracy = 1/3 (chance level), accuracy = majority class share.
    """

    def __init__(self):
        self.majority_class_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MajorityClassBaseline':
        counts = np.bincount(y, minlength=3)
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), 3), dtype=np.float32)
        probs[:, self.majority_class_] = 1.0
        return probs


class PersistenceBaseline:
    """
    Predicts the label that the *previous* sample had (last-observation carry-forward).

    For the first test sample the majority training class is used as the fallback.
    This is the trivial momentum / persistence benchmark: if LOB state persists,
    can we just predict yesterday's direction?
    """

    def __init__(self):
        self.fallback_class_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PersistenceBaseline':
        counts = np.bincount(y, minlength=3)
        self.fallback_class_ = int(np.argmax(counts))
        return self

    def predict(self, X: np.ndarray, y_prev: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        X       : feature matrix (not used)
        y_prev  : previous labels, shape (N,). If None, uses a zero-shift of the
                  test sequence (i.e., the predict call must be made with context).
        """
        n = len(X)
        preds = np.full(n, self.fallback_class_, dtype=np.int64)
        if y_prev is not None and len(y_prev) == n:
            preds = y_prev.astype(np.int64)
        return preds

    def predict_proba(self, X: np.ndarray, y_prev: np.ndarray = None) -> np.ndarray:
        preds = self.predict(X, y_prev)
        probs = np.zeros((len(X), 3), dtype=np.float32)
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
        return probs


class LogisticRegressionBaseline:
    """
    L2-regularised multinomial logistic regression.

    Acts as the linear model upper bound — if neural models cannot beat this,
    the representation is not contributing.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, multi_class='multinomial',
            solver='lbfgs', random_state=seed, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LogisticRegressionBaseline':
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class RandomBaseline:
    """
    Seeded uniform random predictions.

    The theoretical chance level: balanced accuracy = 1/3, MCC ≈ 0.
    Variance across seeds tells us the sampling noise floor.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomBaseline':
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._rng.integers(0, 3, size=len(X)).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._rng.random((len(X), 3)).astype(np.float32)
        return raw / raw.sum(axis=1, keepdims=True)


# Mapping for use in run_all / train_tree pipeline
BASELINE_MAP = {
    'majority_class': MajorityClassBaseline,
    'persistence':    PersistenceBaseline,
    'logistic_regression': LogisticRegressionBaseline,
    'random':         RandomBaseline,
}


def build_baseline(name: str, seed: int = 42):
    """Factory: returns an instantiated (unfitted) baseline by name."""
    if name not in BASELINE_MAP:
        raise ValueError(f"Unknown baseline {name!r}. Options: {list(BASELINE_MAP)}")
    cls = BASELINE_MAP[name]
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()
