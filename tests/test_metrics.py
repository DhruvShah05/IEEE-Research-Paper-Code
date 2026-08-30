"""
tests/test_metrics.py — Sanity checks for eval/metrics.py with hand-computed known cases.

build.md §6 requires known-input / known-output tests for every metric in compute_all_metrics().
All expected values below are hand-calculated and annotated so failures are easy to diagnose.
"""

import pytest
import numpy as np
from eval.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    return y, y.copy()


@pytest.fixture
def balanced_half_wrong():
    """2 samples per class, 1 correct per class → accuracy = 0.5."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])   # exactly 1 wrong per class
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Perfect prediction
# ---------------------------------------------------------------------------

def test_perfect_accuracy(perfect):
    y_true, y_pred = perfect
    res = compute_all_metrics(y_true, y_pred)
    assert res['accuracy'] == 1.0


def test_perfect_balanced_accuracy(perfect):
    y_true, y_pred = perfect
    res = compute_all_metrics(y_true, y_pred)
    assert res['balanced_accuracy'] == 1.0


def test_perfect_macro_f1(perfect):
    y_true, y_pred = perfect
    res = compute_all_metrics(y_true, y_pred)
    assert res['macro_f1'] == 1.0


def test_perfect_mcc(perfect):
    y_true, y_pred = perfect
    res = compute_all_metrics(y_true, y_pred)
    assert res['mcc'] == 1.0


def test_perfect_per_class_f1(perfect):
    y_true, y_pred = perfect
    res = compute_all_metrics(y_true, y_pred)
    assert res['f1_down'] == 1.0
    assert res['f1_stationary'] == 1.0
    assert res['f1_up'] == 1.0


# ---------------------------------------------------------------------------
# Half-wrong (balanced, 1 error per class)
# ---------------------------------------------------------------------------

def test_imperfect_accuracy(balanced_half_wrong):
    y_true, y_pred = balanced_half_wrong
    res = compute_all_metrics(y_true, y_pred)
    assert res['accuracy'] == pytest.approx(0.5)


def test_imperfect_macro_f1(balanced_half_wrong):
    """
    For each class (2 true, 2 predicted, 1 correct):
      precision = recall = 0.5  →  F1 = 0.5
    Macro-F1 = 0.5
    """
    y_true, y_pred = balanced_half_wrong
    res = compute_all_metrics(y_true, y_pred)
    assert res['macro_f1'] == pytest.approx(0.5, abs=1e-6)


def test_imperfect_balanced_accuracy(balanced_half_wrong):
    """
    Recall per class: class 0 → 0.5, class 1 → 0.5, class 2 → 0.5
    Balanced accuracy = mean(0.5, 0.5, 0.5) = 0.5
    """
    y_true, y_pred = balanced_half_wrong
    res = compute_all_metrics(y_true, y_pred)
    assert res['balanced_accuracy'] == pytest.approx(0.5, abs=1e-6)


def test_imperfect_per_class_f1(balanced_half_wrong):
    """Each class has precision=recall=0.5, so per-class F1 = 0.5."""
    y_true, y_pred = balanced_half_wrong
    res = compute_all_metrics(y_true, y_pred)
    assert res['f1_down']        == pytest.approx(0.5, abs=1e-6)
    assert res['f1_stationary']  == pytest.approx(0.5, abs=1e-6)
    assert res['f1_up']          == pytest.approx(0.5, abs=1e-6)


def test_imperfect_weighted_f1(balanced_half_wrong):
    """
    Balanced classes (2 per class) + equal F1 per class → weighted_f1 == macro_f1 = 0.5.
    """
    y_true, y_pred = balanced_half_wrong
    res = compute_all_metrics(y_true, y_pred)
    assert res['weighted_f1'] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# MCC with a non-trivial known case
# ---------------------------------------------------------------------------

def test_mcc_non_trivial():
    """
    3 classes, 6 samples.
    y_true: [0, 0, 1, 1, 2, 2]
    y_pred: [0, 0, 1, 1, 2, 0]   ← last sample misclassified (2→0)

    Hand-computed confusion matrix:
         pred 0  pred 1  pred 2
    true 0:  2       0       0
    true 1:  0       2       0
    true 2:  1       0       1

    sklearn's matthews_corrcoef should give a value strictly in (0, 1),
    clearly above 0 (better than random) but below 1 (not perfect).
    """
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 0])
    res = compute_all_metrics(y_true, y_pred)
    assert 0.0 < res['mcc'] < 1.0, f"Expected MCC in (0,1), got {res['mcc']}"


# ---------------------------------------------------------------------------
# Confusion matrix spot-checks
# ---------------------------------------------------------------------------

def test_confusion_matrix_entries():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])   # class 2 misclassified as 1

    res = compute_all_metrics(y_true, y_pred)
    cm = res['confusion_matrix']

    assert cm[0][0] == 1, "class 0 correctly predicted"
    assert cm[1][1] == 1, "class 1 correctly predicted"
    assert cm[2][1] == 1, "class 2 predicted as class 1"
    assert cm[2][2] == 0, "class 2 never predicted as itself"


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 0, 2, 1])
    res = compute_all_metrics(y_true, y_pred)
    cm = res['confusion_matrix']
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


# ---------------------------------------------------------------------------
# All metric keys are present
# ---------------------------------------------------------------------------

def test_output_keys():
    y = np.array([0, 1, 2])
    res = compute_all_metrics(y, y)
    expected_keys = {
        'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1',
        'f1_down', 'f1_stationary', 'f1_up', 'mcc', 'confusion_matrix'
    }
    assert expected_keys.issubset(res.keys()), f"Missing keys: {expected_keys - res.keys()}"

