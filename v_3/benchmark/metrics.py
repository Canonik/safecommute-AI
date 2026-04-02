"""Classification quality metrics for benchmarking."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute all classification metrics from true labels and predicted probabilities.

    Returns a dict with: accuracy, precision, recall, f1, auc_roc,
    confusion_matrix, optimal_threshold, and per-class metrics.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    prec_per, rec_per, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    # Find optimal threshold (F1-maximizing)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        f1_scores = []
        for t in thresholds:
            p = (y_prob >= t).astype(int)
            _, _, f1_t, _ = precision_recall_fscore_support(
                y_true, p, average='weighted', zero_division=0)
            f1_scores.append(f1_t)
        opt_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[opt_idx])
        optimal_f1 = float(f1_scores[opt_idx])
    except Exception:
        optimal_threshold = 0.5
        optimal_f1 = float(f1)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist(),
        'threshold_used': threshold,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'per_class': {
            'safe': {
                'precision': float(prec_per[0]),
                'recall': float(rec_per[0]),
                'f1': float(f1_per[0]),
                'support': int(sup[0]),
            },
            'unsafe': {
                'precision': float(prec_per[1]) if len(prec_per) > 1 else 0.0,
                'recall': float(rec_per[1]) if len(rec_per) > 1 else 0.0,
                'f1': float(f1_per[1]) if len(f1_per) > 1 else 0.0,
                'support': int(sup[1]) if len(sup) > 1 else 0,
            },
        },
    }
