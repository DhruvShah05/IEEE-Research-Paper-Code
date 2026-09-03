"""
eval/metrics.py — Standardised metric computation for all models and both markets.

Changes (4.1):
  - Added per-class precision and recall.
  - Added Cohen's κ (kappa).
  - Added log-loss and Brier score (require probabilities → optional).
  - Added expected calibration error (ECE).
  - Added "lift over majority-class" (accuracy − majority_class_share) so that
    chance level is visible in the table without needing a separate baseline row.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    cohen_kappa_score,
    log_loss,
    precision_score,
    recall_score,
    brier_score_loss,
)


def _expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray,
                                 n_bins: int = 10) -> float:
    """
    Multi-class Expected Calibration Error (ECE).

    For each class: bin predicted probabilities; measure |mean_conf − mean_acc|
    weighted by bin size.  Averaged over classes.
    """
    n_classes = y_proba.shape[1]
    ece_per_class = []
    for c in range(n_classes):
        prob_c  = y_proba[:, c]
        label_c = (y_true == c).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece_c = 0.0
        for i in range(n_bins):
            mask = (prob_c >= bins[i]) & (prob_c < bins[i + 1])
            if mask.sum() == 0:
                continue
            mean_conf = prob_c[mask].mean()
            mean_acc  = label_c[mask].mean()
            ece_c += mask.sum() * abs(mean_conf - mean_acc)
        ece_per_class.append(ece_c / len(y_true))
    return float(np.mean(ece_per_class))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> dict:
    """
    Standardised metric computation.
    Every model across both markets uses this function to evaluate predictions.

    Parameters
    ----------
    y_true  : (N,)   integer ground-truth labels (0=Down, 1=Stat, 2=Up)
    y_pred  : (N,)   integer predicted labels
    y_proba : (N, 3) predicted class probabilities (optional).
              Required for log_loss, brier_score, and ECE.

    Returns
    -------
    dict of metric name → scalar value (JSON-serialisable).
    """
    labels = [0, 1, 2]

    acc    = accuracy_score(y_true, y_pred)
    bacc   = balanced_accuracy_score(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred, labels=labels)
    macro_f1   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)

    # --- Per-class F1, precision, recall ---
    per_class_f1   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_rec  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)

    def _safe(arr, idx):
        return float(arr[idx]) if idx < len(arr) else 0.0

    # --- Majority class share and lift ---
    counts = np.bincount(y_true, minlength=3)
    majority_share = float(counts.max() / max(len(y_true), 1))
    lift = float(acc) - majority_share

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    result = {
        # Core metrics
        "accuracy":            float(acc),
        "balanced_accuracy":   float(bacc),
        "macro_f1":            float(macro_f1),
        "weighted_f1":         float(weighted_f1),
        "mcc":                 float(mcc),
        "kappa":               float(kappa),
        # Per-class (convention: index 0=Down, 1=Stat, 2=Up)
        "f1_down":             _safe(per_class_f1,   0),
        "f1_stationary":       _safe(per_class_f1,   1),
        "f1_up":               _safe(per_class_f1,   2),
        "precision_down":      _safe(per_class_prec, 0),
        "precision_stationary": _safe(per_class_prec, 1),
        "precision_up":        _safe(per_class_prec, 2),
        "recall_down":         _safe(per_class_rec,  0),
        "recall_stationary":   _safe(per_class_rec,  1),
        "recall_up":           _safe(per_class_rec,  2),
        # Chance-level reference
        "majority_class_share": majority_share,
        "lift_over_majority":   lift,
        # Confusion matrix
        "confusion_matrix":    cm,
    }

    # --- Probability-based metrics (optional) ---
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        # Clip for numerical stability
        y_proba = np.clip(y_proba, 1e-9, 1.0 - 1e-9)

        # log_loss
        result["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))

        # Brier score: mean over one-vs-rest per class, then averaged
        brier_per_class = []
        for c in range(3):
            binary_labels = (y_true == c).astype(float)
            brier_per_class.append(
                float(brier_score_loss(binary_labels, y_proba[:, c]))
            )
        result["brier_score_down"]       = brier_per_class[0]
        result["brier_score_stationary"] = brier_per_class[1]
        result["brier_score_up"]         = brier_per_class[2]
        result["brier_score_mean"]       = float(np.mean(brier_per_class))

        # ECE
        result["expected_calibration_error"] = _expected_calibration_error(y_true, y_proba)

    return result
