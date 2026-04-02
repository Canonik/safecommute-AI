"""
calibrate_thresholds.py
─────────────────────────────────────────────────────────────────────────────
Standalone threshold calibration script.

Run this any time you want to re-calibrate the amber/red thresholds without
retraining — e.g. after collecting new validation data, or when moving to a
new deployment environment with different ambient noise levels.

Usage:
    python calibrate_thresholds.py
    python calibrate_thresholds.py --fpr 0.02   # stricter: ≤2% FPR for red

Outputs:
    thresholds.json   — loaded by mvp_inference.py automatically
    roc_curve.png     — ROC plot (requires matplotlib)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — must match train_model.py
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "prepared_data"
STATS_PATH      = "feature_stats.json"
THRESHOLDS_PATH = "thresholds.json"
MODEL_PATH      = "safecommute_edge_model.pth"
N_MELS          = 64
TIME_FRAMES     = 188
BATCH_SIZE      = 64


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (copy of train_model.py — must stay in sync)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, pool=(2, 2)):
        return F.avg_pool2d(self.net(x), pool)


class SafeCommuteCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input    = nn.BatchNorm2d(1)
        self.block1      = ConvBlock(1,   64)
        self.block2      = ConvBlock(64,  128)
        self.block3      = ConvBlock(128, 256)
        freq_dim         = n_mels // (2 ** 3)
        self.freq_reduce = nn.Linear(256 * freq_dim, 256)
        self.gru         = nn.GRU(256, 128, batch_first=True)
        self.dropout     = nn.Dropout(0.3)
        self.fc          = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))
        _, h = self.gru(x)
        return self.fc(self.dropout(h.squeeze(0)))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET (minimal copy — only needs val split)
# ─────────────────────────────────────────────────────────────────────────────
class ValDataset(Dataset):
    def __init__(self, val_dir, mean=0.0, std=1.0):
        self.files  = []
        self.labels = []
        self.mean   = mean
        self.std    = std
        for lbl, name in enumerate(['0_safe', '1_unsafe']):
            folder = os.path.join(val_dir, name)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith('.pt'):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(lbl)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat = torch.load(self.files[idx], weights_only=True)
        t = feat.shape[-1]
        if t > TIME_FRAMES:
            feat = feat[:, :, :TIME_FRAMES]
        elif t < TIME_FRAMES:
            feat = torch.cat([feat, torch.zeros(1, feat.shape[1], TIME_FRAMES - t)], dim=-1)
        feat = (feat - self.mean) / (self.std + 1e-8)
        return feat, torch.tensor(self.labels[idx], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(target_fpr: float = 0.05):
    # ── Load stats ────────────────────────────────────────────────────────
    if not os.path.exists(STATS_PATH):
        print(f"Error: '{STATS_PATH}' not found. Run data_pipeline.py first.")
        sys.exit(1)
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run train_model.py first.")
        sys.exit(1)
    device = torch.device("cpu")
    model  = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    # ── Collect probabilities ─────────────────────────────────────────────
    val_dir    = os.path.join(DATA_DIR, 'val')
    dataset    = ValDataset(val_dir, mean, std)
    if len(dataset) == 0:
        print("Error: No validation data found.")
        sys.exit(1)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for inp, lab in loader:
            probs = torch.softmax(model(inp.to(device)), dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(lab.tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)

    print(f"\nValidation set: {len(labels_arr)} samples  "
          f"({(labels_arr==0).sum()} safe, {(labels_arr==1).sum()} unsafe)")

    # ── ROC curve ─────────────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = roc_curve(labels_arr, probs_arr)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # RED threshold: highest threshold where FPR ≤ target_fpr
    valid_mask = fpr <= target_fpr
    if valid_mask.any():
        idx_red   = np.where(valid_mask)[0][-1]
        red_thresh = float(roc_thresholds[idx_red])
        red_fpr    = float(fpr[idx_red])
        red_tpr    = float(tpr[idx_red])
    else:
        red_thresh, red_fpr, red_tpr = 0.70, None, None
        print(f"  Warning: no threshold achieves FPR≤{target_fpr}. Using fallback 0.70.")

    # AMBER threshold: highest threshold where FPR ≤ 3× target (less strict)
    amber_fpr_target = min(target_fpr * 3, 0.20)
    valid_amber = fpr <= amber_fpr_target
    if valid_amber.any():
        idx_amb     = np.where(valid_amber)[0][-1]
        amber_thresh = float(roc_thresholds[idx_amb])
    else:
        amber_thresh = max(0.5, red_thresh - 0.15)

    # F1-optimal threshold
    prec, rec, pr_thresholds = precision_recall_curve(labels_arr, probs_arr)
    f1_scores  = 2 * prec * rec / (prec + rec + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])  # last entry has no corresponding threshold
    f1_thresh   = float(pr_thresholds[best_f1_idx])
    best_f1     = float(f1_scores[best_f1_idx])

    print(f"\nCalibrated thresholds (target FPR ≤ {target_fpr*100:.0f}%):")
    print(f"  Amber : {amber_thresh:.3f}  (FPR ≤ {amber_fpr_target*100:.0f}%)")
    if red_fpr is not None:
        print(f"  Red   : {red_thresh:.3f}  (actual FPR={red_fpr:.3f}, TPR={red_tpr:.3f})")
    else:
        print(f"  Red   : {red_thresh:.3f}  (fallback)")
    print(f"  F1-opt: {f1_thresh:.3f}  (F1={best_f1:.3f})")

    # Operational estimate: false alarms per minute at 1 stride/second
    if red_fpr is not None:
        safe_count  = (labels_arr == 0).sum()
        fa_rate_per_min = red_fpr * safe_count / max(len(labels_arr), 1) * 60
        print(f"\n  Estimated false alerts in normal operation: "
              f"~{fa_rate_per_min:.1f} per minute at red threshold")
        print(f"  (This is an approximation based on val-set class balance.)")

    # ── Save thresholds ───────────────────────────────────────────────────
    result = {
        "amber":      round(amber_thresh, 3),
        "red":        round(red_thresh, 3),
        "f1_optimal": round(f1_thresh, 3),
        "roc_auc":    round(roc_auc, 4),
        "note":       (f"Calibrated at target_fpr={target_fpr}. "
                       f"Red: FPR≤{target_fpr*100:.0f}% on val set.")
    }
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {THRESHOLDS_PATH}")

    # ── ROC plot ──────────────────────────────────────────────────────────
    try:
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
                        help='Target false positive rate for the RED threshold (default: 0.05)')
    args = parser.parse_args()
    main(target_fpr=args.fpr)