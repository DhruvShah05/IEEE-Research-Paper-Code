import numpy as np
from scipy.stats import binom

def paired_bootstrap(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, metric_fn, n_boot: int = 2000):
    """
    Paired bootstrap test for difference in performance.
    metric_fn should take (y_true, preds) and return a scalar metric.
    Returns (mean_diff, ci_low, ci_high, p_value).
    """
    n = len(y_true)
    diffs = np.zeros(n_boot)
    
    # We do a vectorized bootstrap for speed if possible, or just a loop
    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        metric_a = metric_fn(y_true[idx], preds_a[idx])
        metric_b = metric_fn(y_true[idx], preds_b[idx])
        diffs[i] = metric_a - metric_b
        
    mean_diff = np.mean(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    
    # p-value: proportion of times the difference is less than 0 (assuming we test if A > B)
    # 2-sided test
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    
    return mean_diff, ci_low, ci_high, p_value

def mcnemar(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray):
    """
    Exact McNemar's test for comparing classification models.
    """
    # Find where predictions were correct
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    
    # A correct, B wrong
    n01 = np.sum(correct_a & ~correct_b)
    # A wrong, B correct
    n10 = np.sum(~correct_a & correct_b)
    
    if n01 + n10 == 0:
        return 1.0 # P-value is 1 if they make exactly the same errors
        
    # Exact binomial test
    p_value = binom.cdf(min(n01, n10), n01 + n10, 0.5) * 2
    return min(p_value, 1.0)
