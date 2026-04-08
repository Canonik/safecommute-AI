"""
Deployment reliability gates for SafeCommute AI.

Evaluates a model against non-negotiable deployment KPIs. Every gate must
pass before a model can be considered production-ready.

Unlike test_deployment.py (which checks latency, size, consistency),
this script checks CLASSIFICATION QUALITY on real-world failure modes:
speech, laughter, crowd, metro ambient.

Usage:
    PYTHONPATH=. python safecommute/pipeline/eval_deployment.py
    PYTHONPATH=. python safecommute/pipeline/eval_deployment.py --model models/cycle6_noise_inject.pth
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH


# ─── DEPLOYMENT GATES ───────────────────────────────────────────────
# These are NON-NEGOTIABLE. A model that fails ANY gate is NOT deployable.

GATES = {
    # GATE 1: Hard-negative FPR ceiling (safe samples misclassified as threat)
    'speech_fpr_max': 0.30,       # speech FPR must be < 30%
    'laughter_fpr_max': 0.40,     # laughter FPR must be < 40%
    'crowd_fpr_max': 0.40,        # crowd FPR must be < 40%
    'metro_fpr_max': 0.20,        # metro ambient FPR must be < 20%

    # GATE 2: Threat recall floor (unsafe samples correctly detected)
    'scream_recall_min': 0.75,    # scream recall must be > 75%
    'shout_recall_min': 0.60,     # shout recall must be > 60%
    'yell_recall_min': 0.75,      # yell recall must be > 75%

    # GATE 3: Worst-source floor
    'worst_source_acc_min': 0.20, # no source below 20% accuracy

    # GATE 4: Latency
    'latency_mean_max_ms': 15.0,  # mean inference < 15ms
    'latency_p99_max_ms': 30.0,   # p99 inference < 30ms
}


def get_source_prefix(filename):
    """Extract source prefix from .pt filename for per-source evaluation."""
    base = os.path.basename(filename).replace('.pt', '')
    if base.startswith('as_'):
        parts = base.split('_')
        if len(parts) >= 2:
            return f"as_{parts[1]}"
    elif base.startswith('yt_metro'):
        return 'yt_metro'
    elif base.startswith('yt_scream'):
        return 'yt_scream'
    elif base.startswith('viol_'):
        return 'violence'
    elif base.startswith('esc_'):
        return 'esc'
    elif base.startswith('bg_') or base.startswith('hns_'):
        return 'hns'
    return 'unknown'


def evaluate_model(model_path, threshold=0.5, device='cpu'):
    """Run full deployment evaluation."""

    # Load model
    device = torch.device(device)
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load stats
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            s = json.load(f)
        mean, std = s['mean'], s['std']
    else:
        mean, std = 0.0, 1.0

    # Load test set
    test_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Collect predictions with source info
    all_probs = []
    all_labels = []
    source_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'probs': [], 'labels': []})

    sample_idx = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            labs = labels.tolist()

            for p, l in zip(probs, labs):
                all_probs.append(p)
                all_labels.append(l)

                # Per-source tracking
                if sample_idx < len(test_ds.filepaths):
                    src = get_source_prefix(test_ds.filepaths[sample_idx])
                    pred = 1 if p > threshold else 0
                    source_results[src]['correct'] += int(pred == l)
                    source_results[src]['total'] += 1
                    source_results[src]['probs'].append(p)
                    source_results[src]['labels'].append(l)
                sample_idx += 1

    # Global metrics
    auc = roc_auc_score(all_labels, all_probs)
    preds = [1 if p > threshold else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='weighted')

    # Per-source accuracy
    source_acc = {}
    for src, data in sorted(source_results.items()):
        source_acc[src] = data['correct'] / max(data['total'], 1)

    # Compute FPR for safe sources (label=0 samples predicted as unsafe)
    def compute_fpr(src_prefix):
        data = source_results.get(src_prefix)
        if not data or not data['labels']:
            return None
        safe_preds = [1 if p > threshold else 0 for p, l in zip(data['probs'], data['labels']) if l == 0]
        if not safe_preds:
            return None
        return sum(safe_preds) / len(safe_preds)

    # Compute recall for unsafe sources (label=1 samples correctly detected)
    def compute_recall(src_prefix):
        data = source_results.get(src_prefix)
        if not data or not data['labels']:
            return None
        unsafe_correct = sum(1 for p, l in zip(data['probs'], data['labels']) if l == 1 and p > threshold)
        unsafe_total = sum(1 for l in data['labels'] if l == 1)
        if unsafe_total == 0:
            return None
        return unsafe_correct / unsafe_total

    # Latency benchmark
    dummy = torch.randn(1, 1, 64, 188).to(device)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)
    lat_mean = np.mean(latencies)
    lat_p99 = np.percentile(latencies, 99)

    # Compute gate results
    results = {
        'global': {'auc': auc, 'accuracy': acc, 'f1': f1, 'threshold': threshold},
        'per_source_accuracy': source_acc,
        'latency': {'mean_ms': lat_mean, 'p99_ms': lat_p99},
    }

    # Gate checks
    gates_passed = 0
    gates_total = 0
    gate_details = []

    def check_gate(name, value, limit, mode='max'):
        nonlocal gates_passed, gates_total
        gates_total += 1
        if value is None:
            gate_details.append(f"  [ SKIP ] {name}: no data")
            return
        if mode == 'max':
            passed = value <= limit
        else:  # min
            passed = value >= limit
        gates_passed += int(passed)
        status = " PASS " if passed else " FAIL "
        gate_details.append(f"  [{status}] {name}: {value:.3f} (limit: {limit:.3f})")

    # GATE 1: Hard-negative FPR
    check_gate('speech_fpr', compute_fpr('as_speech'), GATES['speech_fpr_max'], 'max')
    check_gate('laughter_fpr', compute_fpr('as_laughter'), GATES['laughter_fpr_max'], 'max')
    check_gate('crowd_fpr', compute_fpr('as_crowd'), GATES['crowd_fpr_max'], 'max')
    check_gate('metro_fpr', compute_fpr('yt_metro'), GATES['metro_fpr_max'], 'max')

    # GATE 2: Threat recall
    check_gate('scream_recall', compute_recall('as_screaming'), GATES['scream_recall_min'], 'min')
    check_gate('shout_recall', compute_recall('as_shout'), GATES['shout_recall_min'], 'min')
    check_gate('yell_recall', compute_recall('as_yell'), GATES['yell_recall_min'], 'min')

    # GATE 3: Worst-source floor
    worst_src = min(source_acc.values()) if source_acc else 0
    worst_name = min(source_acc, key=source_acc.get) if source_acc else 'none'
    check_gate(f'worst_source ({worst_name})', worst_src, GATES['worst_source_acc_min'], 'min')

    # GATE 4: Latency
    check_gate('latency_mean', lat_mean, GATES['latency_mean_max_ms'], 'max')
    check_gate('latency_p99', lat_p99, GATES['latency_p99_max_ms'], 'max')

    # Print results
    print("=" * 60)
    print(" DEPLOYMENT RELIABILITY GATES")
    print("=" * 60)
    print(f"\n  Model: {model_path}")
    print(f"  Threshold: {threshold}")
    print(f"  AUC: {auc:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")
    print(f"\n  Per-source accuracy:")
    for src, a in sorted(source_acc.items(), key=lambda x: x[1]):
        n = source_results[src]['total']
        print(f"    {src:20s} {a:.1%}  ({n} samples)")

    print(f"\n  Gate Results ({gates_passed}/{gates_total}):")
    for line in gate_details:
        print(line)

    deployable = gates_passed == gates_total
    print(f"\n  {'DEPLOYABLE' if deployable else 'NOT DEPLOYABLE'}")
    print("=" * 60)

    return deployable, results


def main():
    parser = argparse.ArgumentParser(description='Deployment reliability gates')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    deployable, _ = evaluate_model(args.model, args.threshold, args.device)
    sys.exit(0 if deployable else 1)


if __name__ == "__main__":
    main()
