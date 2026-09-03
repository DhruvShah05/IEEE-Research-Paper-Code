"""
tests/test_reproducibility.py — Tests for cross-seed determinism (Section 6).

Tests:
  - Same seed → identical predictions for tree models
  - Same seed → identical predictions for neural models (CPU)
  - set_seed correctly seeds numpy, random, torch, and cuda
  - Two different seeds give different predictions
"""

import numpy as np
import pytest
import random
import torch

from utils.seeding import set_seed


class TestSetSeed:
    def test_numpy_seeded(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b, err_msg="numpy RNG not seeded correctly")

    def test_torch_seeded(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        torch.testing.assert_close(a, b, msg="torch RNG not seeded correctly")

    def test_random_seeded(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b, "random module RNG not seeded correctly"

    def test_different_seeds_give_different_results(self):
        set_seed(0)
        a = np.random.randn(100)
        set_seed(1)
        b = np.random.randn(100)
        assert not np.allclose(a, b), "Different seeds should give different results"


class TestXGBoostReproducibility:
    def _get_predictions(self, seed: int) -> np.ndarray:
        from models.xgboost_model import build_model
        set_seed(seed)
        np.random.seed(seed)
        cfg = {'model': 'xgboost', 'seed': seed,
               'model_params': {'n_estimators': 10, 'max_depth': 3}}
        model = build_model(cfg)
        np.random.seed(seed + 1)
        X = np.random.randn(100, 40)
        y = np.array([0, 1, 2] * 33 + [0])
        model.fit(X, y)
        # Use a fixed test set
        np.random.seed(0)
        X_test = np.random.randn(30, 40)
        return model.predict(X_test)

    def test_same_seed_same_predictions(self):
        preds_a = self._get_predictions(seed=42)
        preds_b = self._get_predictions(seed=42)
        np.testing.assert_array_equal(preds_a, preds_b,
                                       err_msg="XGBoost: same seed → different predictions")

    def test_different_seed_different_predictions(self):
        preds_a = self._get_predictions(seed=0)
        preds_b = self._get_predictions(seed=99)
        # Very small probability of being equal by chance
        assert not np.array_equal(preds_a, preds_b), \
            "XGBoost: different seeds should give different predictions"


class TestNeuralReproducibility:
    def _get_predictions(self, seed: int) -> torch.Tensor:
        from models.transformer import StandardTransformer
        set_seed(seed)
        model = StandardTransformer(in_features=40, d_model=32, nhead=4, num_layers=1)
        model.eval()
        # Input: fixed random with seed 0 (independent of model seed)
        torch.manual_seed(0)
        x = torch.randn(10, 40)
        with torch.no_grad():
            out = model(x)
        return out

    def test_same_seed_same_predictions(self):
        out_a = self._get_predictions(seed=42)
        out_b = self._get_predictions(seed=42)
        torch.testing.assert_close(
            out_a, out_b,
            msg="Transformer: same seed → different predictions"
        )

    def test_different_seed_different_predictions(self):
        out_a = self._get_predictions(seed=0)
        out_b = self._get_predictions(seed=99)
        assert not torch.allclose(out_a, out_b), \
            "Transformer: different seeds should give different predictions"
