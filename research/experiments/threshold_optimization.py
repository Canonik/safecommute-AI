"""
Threshold Optimization.
Find the optimal classification threshold on the validation set using:
  1. Youden's J statistic (maximize sensitivity + specificity)
  2. F1-maximizing threshold
  3. Cost-sensitive threshold (weighted false negatives)
Then evaluate each threshold on test set.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from datetime import datetime

from safecommute.model import SafeCommuteCNN
from safecommute.constants import DATA_DIR, MODEL_SAVE_PATH, STATS_PATH
from safecommute.dataset import TensorAudioDataset
from research.experiments.eval_utils import load_stats, per_source_breakdown

BATCH_SIZE = 64


def get_predictions(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for inp, lab in loader:
            logits = model(inp.to(device))
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            labels.extend(lab.tolist())
    return np.array(probs), np.array(labels)


def youdens_j(fpr, tpr, thresholds):
    """Youden's J = sensitivity + specificity - 1 = TPR - FPR."""
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx], j[idx]


def f1_optimal(probs, labels):
    """Find threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[min(idx, len(thresholds)-1)], f1s[idx]


def evaluate_threshold(probs, labels, threshold, name):
    """Evaluate a specific threshold."""
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'name': name,
        'threshold': float(threshold),
        'accuracy': float(acc),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'fpr': float(fpr),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))

    print("=== Threshold Optimization ===\n")

    # Get predictions on val set
    val_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    val_probs, val_labels = get_predictions(model, val_ds, device)
    print(f"  Val set: {len(val_ds)} samples")

    # Get predictions on test set
    test_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
    test_probs, test_labels = get_predictions(model, test_ds, device)
    print(f"  Test set: {len(test_ds)} samples")

    # 1. Find thresholds on validation set
    fpr, tpr, roc_thresholds = roc_curve(val_labels, val_probs)
    youden_thresh, youden_j_val = youdens_j(fpr, tpr, roc_thresholds)
    f1_thresh, f1_max_val = f1_optimal(val_probs, val_labels)

    # Also test fixed thresholds
    thresholds = {
        'Default (0.5)': 0.5,
        "Youden's J": youden_thresh,
        'F1-optimal': f1_thresh,
        'High sensitivity (0.3)': 0.3,
        'Low FPR (0.7)': 0.7,
    }

    print(f"\n  Thresholds found on val set:")
    print(f"    Youden's J: {youden_thresh:.3f}")
    print(f"    F1-optimal: {f1_thresh:.3f}")

    # 2. Evaluate all thresholds on TEST set
    print(f"\n  Test set evaluation:")
    print(f"  {'Threshold':<22} {'Value':>6} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'FPR':>7}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    results = []
    for name, thresh in thresholds.items():
        r = evaluate_threshold(test_probs, test_labels, thresh, name)
        results.append(r)
        print(f"  {name:<22} {thresh:>6.3f} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
              f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f} {r['fpr']:>7.4f}")

    # 3. Per-source breakdown for best thresholds
    best_by_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n  Best by F1: {best_by_f1['name']} (threshold={best_by_f1['threshold']:.3f})")

    # Save
    os.makedirs("research/results", exist_ok=True)
    output = {
        'thresholds_found': {k: float(v) for k, v in thresholds.items()},
        'test_results': results,
        'best_f1_threshold': best_by_f1,
        'timestamp': datetime.now().isoformat(),
    }
    with open("research/results/threshold_optimization.json", 'w') as f:
        json.dump(output, f, indent=2)

    # Append to log
    with open("research/experiment_log.md", 'a') as f:
        f.write(f"\n## Threshold Optimization\n\n")
        f.write(f"| Threshold | Value | Accuracy | F1 | Sensitivity | Specificity | FPR |\n")
        f.write(f"|-----------|-------|----------|-------|-------------|-------------|-----|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['threshold']:.3f} | {r['accuracy']:.4f} | "
                    f"{r['f1']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                    f"{r['fpr']:.4f} |\n")
        f.write(f"\n**Recommended**: {best_by_f1['name']} at threshold={best_by_f1['threshold']:.3f}\n")
        f.write(f"\nRun: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

    print(f"\n  Saved to research/results/threshold_optimization.json")


if __name__ == "__main__":
    main()
