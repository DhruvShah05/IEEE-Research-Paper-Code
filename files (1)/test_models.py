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

def make_crypto_layout_row():
    """
    One row in the crypto raw layout with values that encode (side, level, kind):
        bid_p_i = 100*i + 1, bid_v_i = 100*i + 2, ask_p_i = 100*i + 3, ask_v_i = 100*i + 4
    Layout: cols 0..19 = bid_p1,bid_v1,bid_p2,bid_v2,...  cols 20..39 = ask_p1,ask_v1,...
    """
    row = np.zeros(40)
    for i in range(10):
        row[2 * i]      = 100 * (i + 1) + 1   # bid price
        row[2 * i + 1]  = 100 * (i + 1) + 2   # bid vol
        row[20 + 2 * i] = 100 * (i + 1) + 3   # ask price
        row[21 + 2 * i] = 100 * (i + 1) + 4   # ask vol
    return row[None, :]


class TestReorderToCanonical:
    def test_fi2010_unchanged(self):
        from data.loaders import reorder_to_canonical
        X = np.random.randn(50, 144)
        assert np.array_equal(X, reorder_to_canonical(X, 'fi2010'))

    def test_crypto_reorder_shape(self):
        from data.loaders import reorder_to_canonical
        X = np.random.randn(50, 40)
        assert reorder_to_canonical(X, 'crypto').shape == X.shape

    def test_perm_is_a_permutation(self):
        from data.loaders import CRYPTO_TO_CANONICAL_PERM
        assert sorted(CRYPTO_TO_CANONICAL_PERM.tolist()) == list(range(40))

    def test_crypto_reorder_every_level(self):
        """Canonical col 4i..4i+3 must be (ask_p, ask_v, bid_p, bid_v) of level i+1."""
        from data.loaders import reorder_to_canonical
        X_out = reorder_to_canonical(make_crypto_layout_row(), 'crypto')[0]
        for i in range(10):
            base = 100 * (i + 1)
            assert X_out[4 * i + 0] == base + 3, f"level {i+1}: col {4*i} should be ask_price"
            assert X_out[4 * i + 1] == base + 4, f"level {i+1}: col {4*i+1} should be ask_vol"
            assert X_out[4 * i + 2] == base + 1, f"level {i+1}: col {4*i+2} should be bid_price"
            assert X_out[4 * i + 3] == base + 2, f"level {i+1}: col {4*i+3} should be bid_vol"

    def test_prices_and_volumes_never_mix(self):
        """Every canonical price slot must hold a price, every volume slot a volume."""
        from data.loaders import reorder_to_canonical
        X_out = reorder_to_canonical(make_crypto_layout_row(), 'crypto')[0]
        kinds = X_out % 100   # 1/3 = price, 2/4 = volume
        assert np.all(np.isin(kinds[0::2], [1, 3])), "even canonical cols must be prices"
        assert np.all(np.isin(kinds[1::2], [2, 4])), "odd canonical cols must be volumes"

    def test_crypto_extra_cols_passthrough(self):
        from data.loaders import reorder_to_canonical
        X = np.random.randn(10, 50)
        assert np.array_equal(reorder_to_canonical(X, 'crypto')[:, 40:], X[:, 40:])

    def test_reorder_is_invertible(self):
        from data.loaders import reorder_to_canonical, reorder_from_canonical
        X = np.random.randn(5, 40)
        assert np.allclose(reorder_from_canonical(reorder_to_canonical(X, 'crypto'), 'crypto'), X)

    def test_reorder_windowed_3d(self):
        from data.loaders import reorder_to_canonical
        X = np.random.randn(6, 100, 40)
        X_out = reorder_to_canonical(X, 'crypto')
        assert X_out.shape == X.shape
        assert np.array_equal(X_out[:, :, 0], X[:, :, 20])   # ask_p1 comes from crypto col 20

    def test_level_tokenizer_matches_canonical_layout(self):
        """StructuredTransformer 'level' tokens must be (ask_p, ask_v, bid_p, bid_v) per level."""
        from data.loaders import reorder_to_canonical
        from models.structured_transformer import StructuredTransformer
        m = StructuredTransformer(in_features=40, token_mode='level', pooling_mode='mean')
        x = torch.tensor(reorder_to_canonical(make_crypto_layout_row(), 'crypto'), dtype=torch.float32)
        lob = x[:, :40].reshape(1, m.n_lob_tokens, m.group_size)
        assert lob.shape == (1, 10, 4)
        assert torch.equal(lob[0, 0], torch.tensor([103., 104., 101., 102.]))
        assert torch.equal(lob[0, 9], torch.tensor([1003., 1004., 1001., 1002.]))


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


# ---------------------------------------------------------------------------
# Baselines (A5)
# ---------------------------------------------------------------------------

class TestBaselines:
    def _xy(self):
        y = np.array([0, 1, 1, 2, 1, 0, 1, 2, 1, 1])
        return np.zeros((10, 4)), y

    def test_majority_class(self):
        from models.baselines import build_model
        X, y = self._xy()
        m = build_model({'model': 'baseline', 'model_params': {'name': 'majority_class'}}).fit(X, y)
        assert (m.predict(X) == 1).all()
        assert m.predict_proba(X).shape == (10, 3)

    def test_random_predict_matches_proba(self):
        from models.baselines import build_model
        X, y = self._xy()
        m = build_model({'model': 'baseline', 'model_params': {'name': 'random'}, 'seed': 3}).fit(X, y)
        probs = m.predict_proba(X)
        assert np.array_equal(m.predict(X), probs.argmax(axis=1))

    def test_random_is_seeded(self):
        from models.baselines import build_model
        X, y = self._xy()
        a = build_model({'model': 'baseline', 'model_params': {'name': 'random'}, 'seed': 7}).fit(X, y).predict_proba(X)
        b = build_model({'model': 'baseline', 'model_params': {'name': 'random'}, 'seed': 7}).fit(X, y).predict_proba(X)
        assert np.array_equal(a, b)

    def test_persistence_uses_only_past_prices(self):
        from models.baselines import build_model
        X, y = self._xy()
        cfg = {'model': 'baseline', 'model_params': {'name': 'persistence', 'horizon': 2, 'threshold': 1e-4}}
        m = build_model(cfg).fit(X, y)
        mid = np.array([100, 100, 100, 100, 101, 99, 100.02, 100.0, 99.9, 100.5])
        preds = m.predict(X, mid_prices=mid)
        assert preds[4] == 2     # 101 vs mid[2]=100 -> up
        assert preds[5] == 0     # 99  vs mid[3]=100 -> down
        assert preds[6] == 0     # 100.02 vs mid[4]=101 -> down
        assert preds[7] == 2     # 100.0 vs mid[5]=99 -> up
        assert preds[0] == 1 and preds[1] == 1   # no look-back -> fallback (majority = 1)
        # Changing FUTURE prices must not change earlier predictions
        mid2 = mid.copy(); mid2[8:] = 1000.0
        assert np.array_equal(m.predict(X, mid_prices=mid2)[:8], preds[:8])

    def test_persistence_without_mid_falls_back(self):
        from models.baselines import build_model
        X, y = self._xy()
        m = build_model({'model': 'baseline', 'model_params': {'name': 'persistence'}}).fit(X, y)
        assert (m.predict(X) == 1).all()

    def test_logistic_regression(self):
        from models.baselines import build_model
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5)); y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.5).astype(int)
        m = build_model({'model': 'baseline', 'model_params': {'name': 'logistic_regression'},
                         'imbalance': {'strategy': 'class_weight'}}).fit(X, y)
        assert m.predict_proba(X).shape == (300, 3)
        assert (m.predict(X) == y).mean() > 0.6

    def test_build_baseline_factory_rejects_unknown(self):
        from models.baselines import build_model
        with pytest.raises(ValueError):
            build_model({'model': 'baseline', 'model_params': {'name': 'nope'}})
