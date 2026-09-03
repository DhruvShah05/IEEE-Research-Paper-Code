"""
eval/significance.py — Statistical significance tests for model comparisons.

Changes (4.2):
  - Bootstrap is now seeded via np.random.default_rng(seed) — previously
    unseeded, which contradicted the reproducibility claim.
  - Added one_sample_chance_test(): tests whether a model is above chance
    (MCC > 0, Bal-Acc > 1/3) — the most important test given Reviewer 2's
    comment on crypto results.
  - mcnemar and paired_bootstrap signatures unchanged (backward compatible).
"""

import numpy as np
from scipy import stats
from scipy.stats import binom, ttest_rel, wilcoxon


def paired_bootstrap(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple:
    """
    Paired bootstrap test for difference in performance between model A and B.

    Fix 4.2: now seeded via np.random.default_rng(seed).

    Parameters
    ----------
    y_true, preds_a, preds_b : arrays of shape (N,)
    metric_fn : callable (y_true, preds) → scalar
    n_boot    : number of bootstrap replications
    seed      : RNG seed for reproducibility

    Returns
    -------
    (mean_diff, ci_low, ci_high, p_value)
      mean_diff  : mean(metric_A − metric_B) over bootstrap samples
      ci_low     : 2.5th percentile
      ci_high    : 97.5th percentile
      p_value    : two-sided p-value (proportion of diffs on the wrong side)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        metric_a = metric_fn(y_true[idx], preds_a[idx])
        metric_b = metric_fn(y_true[idx], preds_b[idx])
        diffs[i] = metric_a - metric_b

    mean_diff = float(np.mean(diffs))
    ci_low    = float(np.percentile(diffs, 2.5))
    ci_high   = float(np.percentile(diffs, 97.5))
    # Two-sided p-value
    p_value   = float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))

    return mean_diff, ci_low, ci_high, p_value


def bootstrap_ci(
    y_true: np.ndarray,
    preds: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple:
    """
    95 % bootstrap CI for a single model's metric.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], preds[idx])
    return (
        float(metric_fn(y_true, preds)),
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


def mcnemar(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    """
    Exact McNemar's test for comparing classification models.

    Returns the two-sided p-value.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    n01 = int(np.sum( correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a &  correct_b))
    if n01 + n10 == 0:
        return 1.0  # identical error patterns
    p_value = float(binom.cdf(min(n01, n10), n01 + n10, 0.5) * 2)
    return min(p_value, 1.0)


def paired_ttest_across_seeds(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple:
    """
    Paired t-test (two-sided) comparing metric vectors across seeds.

    Parameters
    ----------
    scores_a, scores_b : arrays of shape (n_seeds,)

    Returns
    -------
    (t_stat, p_value)
    """
    t_stat, p_value = ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def wilcoxon_across_seeds(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple:
    """
    Wilcoxon signed-rank test (two-sided) comparing metric vectors across seeds.
    Non-parametric alternative to the paired t-test.

    Returns
    -------
    (stat, p_value)
    """
    try:
        stat, p_value = wilcoxon(scores_a, scores_b, alternative='two-sided')
    except ValueError:
        # Wilcoxon fails when all differences are 0
        stat, p_value = 0.0, 1.0
    return float(stat), float(p_value)


def one_sample_chance_test(
    y_true: np.ndarray,
    preds: np.ndarray,
    metric: str = 'balanced_accuracy',
    chance_level: float = None,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    One-sample test: is the model's metric significantly above chance?

    Fix 4.2: the single most important test for Reviewer 2's comment on crypto
    results (is MCC > 0? is Bal-Acc > 1/3?).

    Parameters
    ----------
    y_true, preds   : arrays of shape (N,)
    metric          : 'balanced_accuracy' (default) or 'mcc'
    chance_level    : null hypothesis value. Defaults to 1/3 for balanced_accuracy,
                      0 for mcc.
    n_boot          : bootstrap replications for CI and p-value

    Returns
    -------
    dict with keys:
        observed, chance_level, ci_low, ci_high, p_value_gt_chance,
        is_above_chance (bool, p < 0.05)
    """
    from eval.metrics import compute_all_metrics

    if metric == 'balanced_accuracy':
        metric_fn = lambda yt, yp: compute_all_metrics(yt, yp)['balanced_accuracy']
        chance = 1 / 3 if chance_level is None else chance_level
    elif metric == 'mcc':
        metric_fn = lambda yt, yp: compute_all_metrics(yt, yp)['mcc']
        chance = 0.0 if chance_level is None else chance_level
    else:
        raise ValueError(f"metric must be 'balanced_accuracy' or 'mcc', got {metric!r}")

    observed = metric_fn(y_true, preds)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = metric_fn(y_true[idx], preds[idx])

    ci_low  = float(np.percentile(boot_scores, 2.5))
    ci_high = float(np.percentile(boot_scores, 97.5))

    # p-value: proportion of bootstrap samples at or below chance
    p_value_gt = float(np.mean(boot_scores <= chance))

    return {
        'metric':           metric,
        'observed':         float(observed),
        'chance_level':     float(chance),
        'ci_low':           ci_low,
        'ci_high':          ci_high,
        'p_value_gt_chance': p_value_gt,
        'is_above_chance':  p_value_gt < 0.05,
    }
