"""
tests/test_backtest.py — Tests for experiments/backtest.py (Section 6).

Tests:
  - Benchmark signals have correct shapes
  - Perfect foresight signal yields positive net PnL
  - Strategy PnL is lower than perfect foresight
  - run_backtest returns structured result dicts
"""

import numpy as np
import os
import json
import pytest


class TestBacktestSignals:
    def test_buy_and_hold_shape(self):
        from experiments.backtest import _buy_and_hold
        n = 100
        sig = _buy_and_hold(n)
        assert sig.shape == (n,)
        assert np.all(sig == 1)

    def test_random_signal_shape(self):
        from experiments.backtest import _random_signal
        sig = _random_signal(100, seed=0)
        assert sig.shape == (100,)
        assert set(sig).issubset({-1, 0, 1})

    def test_majority_signal_shape(self):
        from experiments.backtest import _majority_class_signal
        y = np.array([0, 1, 2, 2, 2, 2, 2])
        sig = _majority_class_signal(y)
        assert sig.shape == (7,)
        assert np.all(sig == 1)  # label 2 (Up) → signal +1

    def test_perfect_foresight_shape(self):
        from experiments.backtest import _perfect_foresight
        y = np.array([0, 1, 2])
        sig = _perfect_foresight(y)
        expected = np.array([-1, 0, 1])  # Down→-1, Stat→0, Up→+1
        np.testing.assert_array_equal(sig, expected)


class TestSimulateStrategy:
    def _make_mid_prices(self, n: int = 200, uptrend: bool = True) -> np.ndarray:
        """Synthetic mid-prices with mild uptrend or flat."""
        if uptrend:
            return 100.0 + np.cumsum(np.abs(np.random.randn(n)) * 0.01)
        return 100.0 + np.random.randn(n) * 0.01

    def test_pf_beats_random_in_uptrend(self):
        """Perfect foresight should outperform random in a deterministic uptrend."""
        from experiments.backtest import _simulate_strategy, _perfect_foresight, _random_signal
        np.random.seed(42)
        n = 500
        mid_prices = self._make_mid_prices(n, uptrend=True)

        # Construct labels from actual returns
        future = np.roll(mid_prices, -10)
        returns = (future - mid_prices) / mid_prices
        y = np.where(returns > 0.0001, 2, np.where(returns < -0.0001, 0, 1))
        y[-10:] = 1

        pf_sig  = _perfect_foresight(y)
        rnd_sig = _random_signal(n, seed=0)

        pf_stats  = _simulate_strategy(pf_sig,  mid_prices, 10, 0, 4.5, 1.0)
        rnd_stats = _simulate_strategy(rnd_sig, mid_prices, 10, 0, 4.5, 1.0)

        assert pf_stats['net_return_bps'] >= rnd_stats['net_return_bps'], \
            "Perfect foresight should not underperform random"

    def test_zero_trades_no_pnl(self):
        """Flat signal = no trades = zero PnL."""
        from experiments.backtest import _simulate_strategy
        mid_prices = self._make_mid_prices(100)
        flat_signal = np.zeros(100, dtype=int)
        stats = _simulate_strategy(flat_signal, mid_prices, 10, 0, 4.5, 1.0)
        assert stats['n_trades'] == 0
        assert stats['net_return_bps'] == 0.0

    def test_keys_present(self):
        """Result dict must contain all expected keys."""
        from experiments.backtest import _simulate_strategy, _buy_and_hold
        mid = self._make_mid_prices(100)
        sig = _buy_and_hold(100)
        stats = _simulate_strategy(sig, mid, 10, 0, 4.5, 1.0)
        for key in ['net_return_bps', 'sharpe', 'max_drawdown_bps',
                    'hit_rate', 'n_trades', 'turnover']:
            assert key in stats, f"Missing key: {key}"


class TestRunBacktest:
    def test_run_backtest_missing_files(self, tmp_path):
        """run_backtest returns empty list when required files are missing."""
        from experiments.backtest import run_backtest
        result = run_backtest(str(tmp_path), horizon=40, out_dir=str(tmp_path))
        assert result == [], "Should return empty list when no probs/labels found"

    def test_run_backtest_with_synthetic_data(self, tmp_path):
        """run_backtest returns structured result dicts with synthetic data."""
        from experiments.backtest import run_backtest
        import json

        n = 500
        np.random.seed(0)
        probs  = np.random.dirichlet([1, 1, 1], size=n).astype(np.float32)
        labels = np.argmax(probs, axis=1)
        mid_prices  = 100.0 + np.cumsum(np.random.randn(n) * 0.01)
        timestamps  = np.arange(n) * 250

        run_dir = str(tmp_path)
        np.save(os.path.join(run_dir, 'test_probs.npy'),       probs)
        np.save(os.path.join(run_dir, 'test_labels.npy'),      labels)
        np.save(os.path.join(run_dir, 'test_mid_prices.npy'),  mid_prices)
        np.save(os.path.join(run_dir, 'test_timestamps.npy'),  timestamps)

        manifest = {'model': 'test_model', 'market': 'crypto'}
        with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as f:
            json.dump(manifest, f)

        results = run_backtest(run_dir, horizon=40, out_dir=str(tmp_path))

        assert len(results) > 0, "run_backtest should produce results with valid inputs"
        for r in results:
            assert 'net_return_bps' in r
            assert 'sharpe' in r
            assert 'n_trades' in r
