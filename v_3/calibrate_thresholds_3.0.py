"""
calibrate_thresholds.py — standalone threshold calibration.

Run this to re-calibrate amber/red thresholds without retraining.

Usage:
    python calibrate_thresholds_3.0.py
    python calibrate_thresholds_3.0.py --fpr 0.02
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import (
    DATA_DIR, STATS_PATH, THRESHOLDS_PATH,
    MODEL_SAVE_PATH as MODEL_PATH,
)

BATCH_SIZE = 64


def main(target_fpr: float = 0.05):
    if not os.path.exists(STATS_PATH):
        print(f"Error: '{STATS_PATH}' not found. Run data_pipeline.py first.")
        sys.exit(1)
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run train_model.py first.")
        sys.exit(1)
    device = torch.device("cpu")
    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    val_dir = os.path.join(DATA_DIR, 'val')
    dataset = TensorAudioDataset(val_dir, mean, std)
    if len(dataset) == 0:
        print("Error: No validation data found.")
        sys.exit(1)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for inp, lab in loader:
            probs = torch.softmax(model(inp.to(device)), dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(lab.tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    print(f"\nValidation set: {len(labels_arr)} samples  "
          f"({(labels_arr==0).sum()} safe, {(labels_arr==1).sum()} unsafe)")

    fpr, tpr, roc_thresholds = roc_curve(labels_arr, probs_arr)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    valid_mask = fpr <= target_fpr
    if valid_mask.any():
        idx_red = np.where(valid_mask)[0][-1]
        red_thresh = float(roc_thresholds[idx_red])
        red_fpr = float(fpr[idx_red])
        red_tpr = float(tpr[idx_red])
    else:
        red_thresh, red_fpr, red_tpr = 0.70, None, None
        print(f"  Warning: no threshold achieves FPR≤{target_fpr}. Using fallback 0.70.")

    amber_fpr_target = min(target_fpr * 3, 0.20)
    valid_amber = fpr <= amber_fpr_target
    if valid_amber.any():
        idx_amb = np.where(valid_amber)[0][-1]
        amber_thresh = float(roc_thresholds[idx_amb])
    else:
        amber_thresh = max(0.5, red_thresh - 0.15)

    prec, rec, pr_thresholds = precision_recall_curve(labels_arr, probs_arr)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    f1_thresh = float(pr_thresholds[best_f1_idx])
    best_f1 = float(f1_scores[best_f1_idx])

    print(f"\nCalibrated thresholds (target FPR ≤ {target_fpr*100:.0f}%):")
    print(f"  Amber : {amber_thresh:.3f}  (FPR ≤ {amber_fpr_target*100:.0f}%)")
    if red_fpr is not None:
        print(f"  Red   : {red_thresh:.3f}  (actual FPR={red_fpr:.3f}, TPR={red_tpr:.3f})")
    else:
        print(f"  Red   : {red_thresh:.3f}  (fallback)")
    print(f"  F1-opt: {f1_thresh:.3f}  (F1={best_f1:.3f})")

    result = {
        "amber": round(amber_thresh, 3),
        "red": round(red_thresh, 3),
        "f1_optimal": round(f1_thresh, 3),
        "roc_auc": round(roc_auc, 4),
        "note": (f"Calibrated at target_fpr={target_fpr}. "
                 f"Red: FPR≤{target_fpr*100:.0f}% on val set.")
    }
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {THRESHOLDS_PATH}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.3f}')
        axes[0].axvline(x=target_fpr, color='r', ls='--', label=f'Target FPR={target_fpr}')
        if red_fpr is not None:
            axes[0].scatter([red_fpr], [red_tpr], color='red', zorder=5, label='Red threshold')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(rec[:-1], prec[:-1], lw=2)
        axes[1].axvline(x=rec[best_f1_idx], color='g', ls='--',
                        label=f'F1-opt threshold={f1_thresh:.2f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.suptitle('SafeCommute AI — Threshold Calibration', fontweight='bold')
        plt.tight_layout()
        plt.savefig('calibration_plots.png', dpi=130)
        print("Plots saved → calibration_plots.png")
        plt.close()
    except ImportError:
        print("(matplotlib not available — plots skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpr', type=float, default=0.05,
                        help='Target false positive rate for RED threshold (default: 0.05)')
    args = parser.parse_args()
    main(target_fpr=args.fpr)
