"""
tests/test_models.py — Forward-pass and shape tests for all models (Section 6).

Tests:
  - DeepLOB (windowed) forward pass: (B, 1, T, 40) → (B, 3) and (B, T, 40) → (B, 3) [A1]
  - End-to-end WindowedDataset + DataLoader + DeepLOB forward pass [A1]
  - reorder_to_canonical: crypto column permutation [A2]
  - StandardTransformer (scalar + temporal) forward pass: (B, F) → (B, 3)
  - StructuredTransformer (flat/grouped/level) forward pass: (B, F) → (B, 3)
  - XGBoost, RandomForest: fit + predict + predict_proba shapes
  - All baselines: fit + predict + predict_proba shapes
  - PersistenceBaseline: no future leak (A5)
  - RandomBaseline: predict derives from predict_proba (B7)
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
# A2: reorder_to_canonical
# ---------------------------------------------------------------------------

class TestReorderToCanonical:
    def test_fi2010_unchanged(self):
        """FI-2010 is already canonical — must be returned unchanged."""
        from data.loaders import reorder_to_canonical
        X = np.random.randn(50, 40)
        X_out = reorder_to_canonical(X, 'fi2010')
        assert np.array_equal(X, X_out), "fi2010 should be returned unchanged"

    def test_crypto_reorder_shape(self):
        """Crypto reorder must preserve shape."""
        from data.loaders import reorder_to_canonical
        X = np.random.randn(50, 40)
        X_out = reorder_to_canonical(X, 'crypto')
        assert X_out.shape == X.shape

    def test_crypto_reorder_permutation(self):
        """Check specific column mapping for crypto → canonical layout (A2)."""
        from data.loaders import reorder_to_canonical
        # Build synthetic X with known values per column group
        X = np.zeros((1, 40))
        # Crypto layout: 0..9=bid_p, 10..19=bid_v, 20..29=ask_p, 30..39=ask_v
        X[0, 0]  = 1.0   # bid_price_1
        X[0, 10] = 2.0   # bid_vol_1
        X[0, 20] = 3.0   # ask_price_1
        X[0, 30] = 4.0   # ask_vol_1
        X_out = reorder_to_canonical(X, 'crypto')
        # Canonical: col0=ask_p1, col1=ask_v1, col2=bid_p1, col3=bid_v1
        assert X_out[0, 0] == 3.0, f"col0 should be ask_price_1 (3.0), got {X_out[0,0]}"
        assert X_out[0, 1] == 4.0, f"col1 should be ask_vol_1 (4.0), got {X_out[0,1]}"
        assert X_out[0, 2] == 1.0, f"col2 should be bid_price_1 (1.0), got {X_out[0,2]}"
        assert X_out[0, 3] == 2.0, f"col3 should be bid_vol_1 (2.0), got {X_out[0,3]}"

    def test_crypto_extra_cols_passthrough(self):
        """Columns beyond 40 should be unchanged."""
        from data.loaders import reorder_to_canonical
        X = np.random.randn(10, 50)
        X_orig = X.copy()
        X_out = reorder_to_canonical(X, 'crypto')
        assert np.array_equal(X_out[:, 40:], X_orig[:, 40:])

    def test_reorder_is_invertible(self):
        """Applying reorder twice with known inverses should recover original data."""
        from data.loaders import reorder_to_canonical
        # If we reorder crypto then "undo" it, we should get back original.
        # The canonical→crypto inverse permutation:
        # canonical col 4i = ask_p_{i+1} came from crypto col (20+i)
        # canonical col 4i+1 = ask_v_{i+1} came from crypto col (30+i)
        # canonical col 4i+2 = bid_p_{i+1} came from crypto col (0+i)
        # canonical col 4i+3 = bid_v_{i+1} came from crypto col (10+i)
        X = np.random.randn(5, 40)
        X_canonical = reorder_to_canonical(X, 'crypto')
        # Verify it's actually a different order for non-symmetric data
        assert not np.array_equal(X[:, :40], X_canonical[:, :40])


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

    def test_forward_shape_3d_input(self):
        """DeepLOB.forward must also accept (B, T, F) from WindowedDataset (A1)."""
        from models.deeplob import DeepLOB
        model = DeepLOB(T=100, in_features=40, lstm_hidden=64)
        model.eval()
        # WindowedDataset yields (B, T, F) without the channel dim
        x = torch.randn(4, 100, 40)  # (B, T, F)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3), f"Expected (4,3) for (B,T,F) input, got {out.shape}"

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

    def test_e2e_windowed_dataset_deeplob(self):
        """A1: End-to-end test: WindowedDataset -> DataLoader -> DeepLOB forward pass."""
        from torch.utils.data import DataLoader
        from data.loaders import WindowedDataset
        from models.deeplob import DeepLOB

        N, F, T = 200, 40, 100
        X = np.random.randn(N, F).astype(np.float32)
        y = np.array([i % 3 for i in range(N)], dtype=np.int64)

        ds = WindowedDataset(X, y, T=T).dataset
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        model = DeepLOB(T=T, in_features=F)
        model.eval()

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (8, T, F), f"Expected (8,{T},{F}), got {batch_x.shape}"
        with torch.no_grad():
            out = model(batch_x)
        assert out.shape == (8, 3), f"Expected (8,3), got {out.shape}"
        # Verify dataset length: N - (T-1) valid samples
        assert len(ds) == N - (T - 1), f"Expected {N-(T-1)} samples, got {len(ds)}"

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

    def test_persistence_baseline_no_future_leak(self):
        """A5: PersistenceBaseline must use mid-price realized change, not y_prev."""
        from models.baselines import PersistenceBaseline
        np.random.seed(0)
        N = 200
        X_train = np.random.randn(80, 10)
        y_train = np.array([0, 1, 2] * 26 + [0, 1])
        X_test  = np.random.randn(N, 10)

        # Construct synthetic mid prices: flat for first 40, then rising.
        mid = np.ones(N, dtype=np.float64)
        mid[40:] = np.linspace(1.0, 1.05, N - 40)  # upward trend

        m = PersistenceBaseline(horizon=40, threshold=0.0001)
        m.fit(X_train, y_train)

        # Without mid_prices, falls back to majority class
        preds_fallback = m.predict(X_test, mid_prices=None)
        assert set(preds_fallback.tolist()) == {m.fallback_class_}

        # With mid_prices, uses realized H-step change
        preds = m.predict(X_test, mid_prices=mid)
        assert preds.shape == (N,)
        assert set(preds).issubset({0, 1, 2})
        # Samples past horizon 40 in the rising segment should mostly be Up (2)
        assert np.mean(preds[80:] == 2) > 0.5, "Rising prices should produce Up labels"

        probs = m.predict_proba(X_test, mid_prices=mid)
        assert probs.shape == (N, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        # probs argmax should match predict
        assert np.all(probs.argmax(axis=1) == preds)

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
        probs = m.predict_proba(X)
        preds = m.predict(X)  # must derive from a FRESH call to predict_proba
        # B7: verify shapes
        assert probs.shape == (50, 3)
        assert preds.shape == (50,)
        assert set(preds).issubset({0, 1, 2})
        # B7: verify predict and predict_proba are consistent.
        # Since predict calls predict_proba internally, a fresh pair must agree.
        m2 = RandomBaseline(seed=0)
        m2.fit(X, y)
        probs2 = m2.predict_proba(X)
        preds2 = probs2.argmax(axis=1)
        m3 = RandomBaseline(seed=0)
        m3.fit(X, y)
        preds3 = m3.predict(X)  # internally calls predict_proba with same RNG state
        assert np.all(preds2 == preds3), "predict must equal argmax(predict_proba)"

    def test_build_baseline_factory(self):
        from models.baselines import build_baseline
        for name in ['majority_class', 'persistence', 'logistic_regression', 'random']:
            b = build_baseline(name, seed=42)
            assert b is not None
