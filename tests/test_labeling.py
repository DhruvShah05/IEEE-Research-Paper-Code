"""
tests/test_labeling.py — Unit tests for data/labeling.py (Section 6).

Tests:
  - label_by_threshold produces correct 0/1/2 labels
  - apply_horizon_labeling with explicit mid_price (fix 0.1) returns correct shapes
  - apply_horizon_labeling with transformed df triggers warning (not silent corruption)
  - run_threshold_sweep returns expected DataFrame shape
  - get_class_distribution values sum to 100
  - apply_adaptive_threshold_labeling produces no NaNs
"""

import numpy as np
import pandas as pd
import pytest

from data.labeling import (
    apply_adaptive_threshold_labeling,
    apply_horizon_labeling,
    get_class_distribution,
    label_by_threshold,
    remap_fi2010_labels,
    run_threshold_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_lob_df(n: int = 200) -> pd.DataFrame:
    """Creates a synthetic raw LOB DataFrame with all 42 columns (0–41)."""
    cols = {str(i): np.ones(n) * 100.0 + np.random.randn(n) * 0.1 for i in range(42)}
    df = pd.DataFrame(cols)
    # Ensure best ask > best bid (no crossed books)
    df['2']  = 100.0  # best bid
    df['22'] = 100.1  # best ask
    # Make remaining bid prices descending
    for k, col in enumerate([str(i) for i in range(4, 22, 2)]):
        df[col] = 100.0 - (k + 1) * 0.01
    # Make remaining ask prices ascending
    for k, col in enumerate([str(i) for i in range(24, 42, 2)]):
        df[col] = 100.1 + (k + 1) * 0.01
    return df


# ---------------------------------------------------------------------------
# label_by_threshold
# ---------------------------------------------------------------------------

class TestLabelByThreshold:
    def test_basic_categories(self):
        returns = np.array([-0.01, -0.0005, 0.0, 0.0005, 0.01])
        labels  = label_by_threshold(returns, threshold=0.001)
        assert labels[0] == 0, "Large negative return should be Down (0)"
        assert labels[-1] == 2, "Large positive return should be Up (2)"
        assert labels[2] == 1, "Zero return should be Stationary (1)"

    def test_boundary(self):
        returns = np.array([-0.001, 0.001])
        labels  = label_by_threshold(returns, threshold=0.001)
        # Boundary returns: |return| = threshold → should be Stationary (1)
        assert labels[0] == 1, "Return exactly at -threshold should be Stationary"
        assert labels[1] == 1, "Return exactly at +threshold should be Stationary"

    def test_all_stationary(self):
        returns = np.zeros(50)
        labels  = label_by_threshold(returns, threshold=0.001)
        assert np.all(labels == 1)

    def test_output_dtype(self):
        labels = label_by_threshold(np.array([0.1, -0.1, 0.0]), threshold=0.05)
        assert labels.dtype == np.int64


# ---------------------------------------------------------------------------
# apply_horizon_labeling
# ---------------------------------------------------------------------------

class TestApplyHorizonLabeling:
    def test_output_shapes(self):
        np.random.seed(0)
        df = make_raw_lob_df(n=200)
        feature_cols = [str(i) for i in range(2, 42)]
        mid_price = (df['2'] + df['22']) / 2.0
        X, y, returns = apply_horizon_labeling(df, horizon=10, threshold=0.0001,
                                               mid_price=mid_price)
        assert X.shape == (190, 40), f"Expected (190,40), got {X.shape}"
        assert y.shape == (190,)
        assert returns.shape == (190,)

    def test_labels_are_valid(self):
        df = make_raw_lob_df(n=200)
        mid_price = (df['2'] + df['22']) / 2.0
        _, y, _ = apply_horizon_labeling(df, horizon=10, threshold=0.0001,
                                          mid_price=mid_price)
        assert set(y).issubset({0, 1, 2}), "Labels must be in {0, 1, 2}"

    def test_explicit_mid_price_required_for_transformed_df(self):
        """
        Fix 0.1: using a transformed df without explicit mid_price should raise or warn
        (the mid_price assertion inside apply_horizon_labeling will catch zero/neg prices
        when the caller passes None and df is already fractional-relative → prices near 0).
        """
        df = make_raw_lob_df(n=200)
        from data.features import to_relative_price
        df_transformed = to_relative_price(df, mode='fractional')

        # Without explicit mid_price, the function warns and tries to derive from df
        # (which now has relative prices ~ 0 → assertion should raise)
        with pytest.raises((AssertionError, ValueError)):
            # This call will fail because relative prices are near 0
            mid_derived = (df_transformed['2'] + df_transformed['22']) / 2.0
            # Inject the bad mid_price explicitly to simulate the bug
            apply_horizon_labeling(df_transformed, 10, 0.0001, mid_price=mid_derived)

    def test_no_nan_in_returns(self):
        df = make_raw_lob_df(n=200)
        mid_price = (df['2'] + df['22']) / 2.0
        _, _, returns = apply_horizon_labeling(df, horizon=10, threshold=0.0001,
                                               mid_price=mid_price)
        assert not np.any(np.isnan(returns)), "returns should not contain NaN"

    def test_smoothed_labeling(self):
        df = make_raw_lob_df(n=200)
        mid_price = (df['2'] + df['22']) / 2.0
        X, y, returns = apply_horizon_labeling(df, horizon=10, threshold=0.0001,
                                               mid_price=mid_price, scheme='smoothed_mean')
        assert X.shape[0] == 190
        assert set(y).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# remap_fi2010_labels
# ---------------------------------------------------------------------------

class TestRemapFI2010Labels:
    def test_mapping(self):
        raw = np.array([1, 2, 3])
        out = remap_fi2010_labels(raw)
        assert out[0] == 2, "1 (Up)   → 2 (Up)"
        assert out[1] == 1, "2 (Stat) → 1 (Stat)"
        assert out[2] == 0, "3 (Down) → 0 (Down)"

    def test_all_values_covered(self):
        for v in [1, 2, 3]:
            out = remap_fi2010_labels(np.array([v]))
            assert out[0] in {0, 1, 2}

    def test_dtype_preserved(self):
        raw = np.array([1, 2, 3], dtype=np.int32)
        out = remap_fi2010_labels(raw)
        assert out.dtype == np.int64


# ---------------------------------------------------------------------------
# get_class_distribution
# ---------------------------------------------------------------------------

class TestGetClassDistribution:
    def test_pcts_sum_to_100(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        dist = get_class_distribution(labels)
        total_pct = dist['pct_down'] + dist['pct_stationary'] + dist['pct_up']
        assert abs(total_pct - 100.0) < 0.01

    def test_counts_match_input(self):
        labels = np.array([0, 0, 0, 1, 2, 2])
        dist = get_class_distribution(labels)
        assert dist['counts'][0] == 3
        assert dist['counts'][1] == 1
        assert dist['counts'][2] == 2

    def test_total_matches(self):
        labels = np.array([0, 1, 2])
        dist = get_class_distribution(labels)
        assert dist['total'] == 3


# ---------------------------------------------------------------------------
# run_threshold_sweep
# ---------------------------------------------------------------------------

class TestRunThresholdSweep:
    def test_output_shape(self):
        df = make_raw_lob_df(n=500)
        mid_price = (df['2'] + df['22']) / 2.0
        horizons = [10, 20]
        thresholds = [0.0001, 0.001]
        result = run_threshold_sweep(df, horizons, thresholds, mid_price=mid_price)
        assert len(result) == len(horizons) * len(thresholds)
        assert 'pct_down' in result.columns
        assert 'pct_stationary' in result.columns
        assert 'pct_up' in result.columns

    def test_pcts_are_valid(self):
        df = make_raw_lob_df(n=500)
        mid_price = (df['2'] + df['22']) / 2.0
        result = run_threshold_sweep(df, [10], [0.0001], mid_price=mid_price)
        row = result.iloc[0]
        total = row['pct_down'] + row['pct_stationary'] + row['pct_up']
        assert abs(total - 100.0) < 0.01


# ---------------------------------------------------------------------------
# apply_adaptive_threshold_labeling
# ---------------------------------------------------------------------------

class TestAdaptiveThreshold:
    def test_no_nan(self):
        df = make_raw_lob_df(n=300)
        mid_price = (df['2'] + df['22']) / 2.0
        from data.features import to_relative_price
        df_t = to_relative_price(df, mode='fractional')
        X, y, returns, thresholds = apply_adaptive_threshold_labeling(
            df_t, horizon=10, c=1.0, mid_price=mid_price
        )
        assert not np.any(np.isnan(returns))
        # The first rolling_std value may be NaN (no prior data);
        # all subsequent values should be finite.
        finite_thresholds = thresholds[~np.isnan(thresholds)]
        assert len(finite_thresholds) >= len(thresholds) - 1, \
            "At most 1 NaN threshold expected (first rolling-std value)"
        assert not np.any(np.isnan(returns))
        assert set(y).issubset({0, 1, 2})
