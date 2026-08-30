import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Standardized metric computation. 
    Every model across both markets uses this function to evaluate predictions.
    """
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_class_0 = float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0
    f1_class_1 = float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0
    f1_class_2 = float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # confusion_matrix returns a 2D array, we convert to list for easy JSON serialization
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "f1_down": f1_class_0,
        "f1_stationary": f1_class_1,
        "f1_up": f1_class_2,
        "mcc": float(mcc),
        "confusion_matrix": cm
    }
