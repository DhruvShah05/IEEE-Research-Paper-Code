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
    Predicts the direction of the last-H realized mid-price change.

    At time t, ``sign(mid[t] - mid[t-H])`` thresholded by ``threshold`` gives:
      2 (Up)   if mid[t] > mid[t-H] + threshold*mid[t-H]
      0 (Down) if mid[t] < mid[t-H] - threshold*mid[t-H]
      1 (Stat) otherwise

    All quantities are available at time t — no future information is used.
    The previous ``y_prev``-based implementation leaked H observations of
    future mid-price information and has been removed (fix A5).
    """

    def __init__(self, horizon: int = 40, threshold: float = 0.0001):
        """
        Parameters
        ----------
        horizon   : look-back H (same as the label horizon, usually 40).
        threshold : fractional return threshold ± for Up/Down classification.
        """
        self.horizon    = horizon
        self.threshold  = threshold
        self.fallback_class_ = None
        self._mid_prices = None  # training mid-prices, kept for warm-start

    def fit(self, X: np.ndarray, y: np.ndarray,
            mid_prices: np.ndarray = None, **kwargs) -> 'PersistenceBaseline':
        """
        Parameters
        ----------
        X          : feature matrix (not used)
        y          : training labels (used only for fallback class)
        mid_prices : 1-D array of raw mid-prices aligned with X.
                     Required for realistic prediction; if None the fallback
                     class is used for all samples.
        """
        counts = np.bincount(y, minlength=3)
        self.fallback_class_ = int(np.argmax(counts))
        if mid_prices is not None:
            self._mid_prices = np.asarray(mid_prices, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray,
                mid_prices: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        X          : feature matrix (not used)
        mid_prices : 1-D array of raw mid-prices for the test window.
                     If provided, uses realized H-step change as the signal.
                     If None, returns the fallback (majority training class).
        """
        n = len(X)
        preds = np.full(n, self.fallback_class_, dtype=np.int64)

        if mid_prices is not None:
            mid = np.asarray(mid_prices, dtype=np.float64)
            H = self.horizon
            thr = self.threshold
            for t in range(n):
                if t < H:
                    preds[t] = self.fallback_class_
                else:
                    ret = (mid[t] - mid[t - H]) / (mid[t - H] + 1e-12)
                    if ret > thr:
                        preds[t] = 2  # Up
                    elif ret < -thr:
                        preds[t] = 0  # Down
                    else:
                        preds[t] = 1  # Stationary
        return preds

    def predict_proba(self, X: np.ndarray,
                      mid_prices: np.ndarray = None) -> np.ndarray:
        preds = self.predict(X, mid_prices)
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._rng.random((len(X), 3)).astype(np.float32)
        return raw / raw.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # B7: derive predict from argmax(predict_proba) so predict and predict_proba agree.
        # NOTE: this advances the RNG state — call predict_proba separately if you need
        # both, or store the result of predict_proba and argmax it yourself.
        probs = self.predict_proba(X)
        return probs.argmax(axis=1).astype(np.int64)


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
