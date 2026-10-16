from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Calculates unweighted macro-F1 and balanced accuracy for multi-class classification."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "n_samples": len(y_true),
    }
