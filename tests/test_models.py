"""
tests/test_models.py — Forward-pass and shape tests for all models (Section 6).

Tests:
  - DeepLOB (windowed) forward pass: (B, 1, T, 40) → (B, 3)
  - StandardTransformer (scalar + temporal) forward pass: (B, F) → (B, 3)
  - StructuredTransformer (flat/grouped/level) forward pass: (B, F) → (B, 3)
  - XGBoost, RandomForest: fit + predict + predict_proba shapes
  - All baselines: fit + predict + predict_proba shapes
  - param_count logging (not NaN)
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_snapshot_batch(B: int, F: int) -> torch.Tensor:
    return torch.randn(B, F)

def make_windowed_batch(B: int, T: int, F: int) -> torch.Tensor:
    return torch.randn(B, 1, T, F)


# ---------------------------------------------------------------------------
# DeepLOB
# ---------------------------------------------------------------------------

class TestDeepLOB:
    def test_forward_shape(self):
        from models.deeplob import DeepLOB
        model = DeepLOB(T=100, in_features=40, lstm_hidden=64)
        model.eval()
        x = make_windowed_batch(B=4, T=100, F=40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3), f"Expected (4,3), got {out.shape}"

    def test_param_count_positive(self):
        from models.deeplob import DeepLOB
        model = DeepLOB(T=100, in_features=40)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_build_model_windowed(self):
        from models.deeplob import build_model
        cfg = {'model': 'deeplob', 'market': 'crypto', 'model_params': {'T': 100}}
        model = build_model(cfg)
        x = make_windowed_batch(B=2, T=100, F=40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_build_model_snapshot(self):
        from models.deeplob import build_model
        cfg = {'model': 'deeplob', 'market': 'crypto',
               'model_params': {'variant': 'snapshot'}}
        model = build_model(cfg)
        model.eval()
        x = torch.randn(4, 40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# StandardTransformer
# ---------------------------------------------------------------------------

class TestStandardTransformer:
    def test_scalar_mode(self):
        from models.transformer import StandardTransformer
        model = StandardTransformer(in_features=40, token_mode='scalar')
        model.eval()
        x = make_snapshot_batch(B=4, F=40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)

    def test_temporal_mode(self):
        from models.transformer import StandardTransformer
        model = StandardTransformer(in_features=40, token_mode='temporal')
        model.eval()
        x = torch.randn(4, 20, 40)  # (B, T, F)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)

    def test_fi2010_features(self):
        from models.transformer import build_model
        cfg = {'market': 'fi2010', 'model_params': {}}
        model = build_model(cfg)
        model.eval()
        x = torch.randn(2, 144)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_configurable_depth(self):
        from models.transformer import build_model
        cfg = {
            'market': 'crypto',
            'model_params': {'d_model': 128, 'nhead': 4, 'num_layers': 4}
        }
        model = build_model(cfg)
        x = torch.randn(2, 40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# StructuredTransformer
# ---------------------------------------------------------------------------

class TestStructuredTransformer:
    @pytest.mark.parametrize("token_mode", ["flat", "grouped", "level"])
    def test_token_modes(self, token_mode):
        from models.structured_transformer import StructuredTransformer
        model = StructuredTransformer(in_features=40, token_mode=token_mode)
        model.eval()
        x = torch.randn(4, 40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3), f"token_mode={token_mode}: expected (4,3), got {out.shape}"

    @pytest.mark.parametrize("pooling_mode", ["mean", "cls", "attention"])
    def test_pooling_modes(self, pooling_mode):
        from models.structured_transformer import StructuredTransformer
        model = StructuredTransformer(in_features=40, pooling_mode=pooling_mode)
        model.eval()
        x = torch.randn(4, 40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3), f"pooling_mode={pooling_mode}: expected (4,3)"

    def test_fi2010_full144(self):
        """StructuredTransformer with FI-2010 144-feature input must work in 'level' mode."""
        from models.structured_transformer import StructuredTransformer
        model = StructuredTransformer(in_features=144, token_mode='level')
        model.eval()
        x = torch.randn(2, 144)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_fi2010_grouped_extra_tokens(self):
        """
        FI-2010 with grouped mode: 40-col LOB block becomes 20 tokens;
        extra 104 derived features become 104 individual tokens.
        Total = 124 tokens.
        """
        from models.structured_transformer import StructuredTransformer
        model = StructuredTransformer(in_features=144, token_mode='grouped')
        assert model.seq_len == 20 + 104, \
            f"Expected 124 tokens for fi2010 grouped mode, got {model.seq_len}"
        model.eval()
        x = torch.randn(2, 144)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_configurable_depth_and_width(self):
        from models.structured_transformer import build_model
        cfg = {
            'market': 'crypto',
            'model_params': {'d_model': 128, 'nhead': 4, 'num_layers': 4,
                             'token_mode': 'level', 'pooling_mode': 'attention'}
        }
        model = build_model(cfg)
        x = torch.randn(2, 40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# Tree Models
# ---------------------------------------------------------------------------

class TestXGBoostModel:
    def test_build_and_predict(self):
        from models.xgboost_model import build_model, get_feature_importances
        cfg = {'model': 'xgboost', 'seed': 42, 'model_params': {
            'n_estimators': 5, 'max_depth': 3
        }}
        model = build_model(cfg)
        X = np.random.randn(50, 40)
        y = np.array([0, 1, 2] * 16 + [0, 1])  # balanced
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert preds.shape == (50,)
        assert probs.shape == (50, 3)
        assert set(preds).issubset({0, 1, 2})

    def test_feature_importances(self):
        from models.xgboost_model import build_model, get_feature_importances
        cfg = {'model': 'xgboost', 'seed': 42, 'model_params': {'n_estimators': 5}}
        model = build_model(cfg)
        X = np.random.randn(30, 40)
        y = np.array([0, 1, 2] * 10)
        model.fit(X, y)
        fi = get_feature_importances(model)
        assert 'gain' in fi
        assert isinstance(fi['gain'], dict)


class TestRandomForestModel:
    def test_build_and_predict(self):
        from models.random_forest import build_model, get_feature_importances
        cfg = {'model': 'random_forest', 'seed': 42,
               'model_params': {'n_estimators': 10},
               'imbalance': {'strategy': 'class_weight'}}
        model = build_model(cfg)
        X = np.random.randn(60, 40)
        y = np.array([0, 1, 2] * 20)
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert preds.shape == (60,)
        assert probs.shape == (60, 3)

    def test_feature_importances(self):
        from models.random_forest import build_model, get_feature_importances
        cfg = {'model': 'random_forest', 'seed': 42,
               'model_params': {'n_estimators': 10}}
        model = build_model(cfg)
        X = np.random.randn(30, 40)
        y = np.array([0, 1, 2] * 10)
        model.fit(X, y)
        fi = get_feature_importances(model)
        assert 'mdi' in fi
        assert len(fi['mdi']) == 40


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_majority_class(self):
        from models.baselines import MajorityClassBaseline
        X = np.random.randn(100, 10)
        y = np.array([0] * 60 + [1] * 30 + [2] * 10)
        m = MajorityClassBaseline()
        m.fit(X, y)
        preds = m.predict(X)
        assert np.all(preds == 0), "Majority class is 0"
        probs = m.predict_proba(X)
        assert probs.shape == (100, 3)
        assert np.all(probs[:, 0] == 1.0)

    def test_logistic_regression(self):
        from models.baselines import LogisticRegressionBaseline
        np.random.seed(0)
        X_train = np.random.randn(90, 40)
        y_train = np.array([0, 1, 2] * 30)
        X_test  = np.random.randn(30, 40)
        m = LogisticRegressionBaseline(seed=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        probs = m.predict_proba(X_test)
        assert preds.shape == (30,)
        assert probs.shape == (30, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_random_baseline(self):
        from models.baselines import RandomBaseline
        X = np.random.randn(50, 10)
        y = np.array([0, 1, 2] * 16 + [0, 1])
        m = RandomBaseline(seed=0)
        m.fit(X, y)
        preds = m.predict(X)
        probs = m.predict_proba(X)
        assert preds.shape == (50,)
        assert set(preds).issubset({0, 1, 2})
        assert probs.shape == (50, 3)

    def test_build_baseline_factory(self):
        from models.baselines import build_baseline
        for name in ['majority_class', 'persistence', 'logistic_regression', 'random']:
            b = build_baseline(name, seed=42)
            assert b is not None
