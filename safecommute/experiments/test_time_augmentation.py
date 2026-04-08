"""
Experiment 2: Test-Time Augmentation (TTA).
Run 3 augmented versions at inference, average softmax outputs.
No retraining needed — free accuracy boost at 3x latency cost.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from datetime import datetime

from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH, DATA_DIR, TIME_FRAMES
from research.experiments.eval_utils import (
    load_stats, get_test_loader, per_source_breakdown,
    count_parameters, model_size_mb, log_experiment
)


def freq_shift(tensor, shift=2):
    """Shift mel spectrogram along frequency axis."""
    shifted = torch.roll(tensor, shifts=shift, dims=-2)
    if shift > 0:
        shifted[:, :, :shift, :] = 0
    elif shift < 0:
        shifted[:, :, shift:, :] = 0
    return shifted


def time_shift(tensor, shift=10):
    """Shift spectrogram along time axis."""
    shifted = torch.roll(tensor, shifts=shift, dims=-1)
    if shift > 0:
        shifted[:, :, :, :shift] = 0
    elif shift < 0:
        shifted[:, :, :, shift:] = 0
    return shifted


def tta_predict(model, inputs, device, n_aug=5):
    """Run TTA: original + freq shifts + time shifts, average softmax."""
    augmented_versions = [
        inputs,                          # original
        freq_shift(inputs, shift=2),     # freq shift up
        freq_shift(inputs, shift=-2),    # freq shift down
        time_shift(inputs, shift=10),    # time shift right
        time_shift(inputs, shift=-10),   # time shift left
    ]

    all_probs = []
    for aug_input in augmented_versions[:n_aug]:
        logits = model(aug_input.to(device))
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    test_dataset, test_loader = get_test_loader(batch_size=32)

    # Evaluate with TTA (3 augmentations)
    for n_aug in [3, 5]:
        print(f"\n=== TTA with {n_aug} augmentations ===")
        all_probs, all_labels, all_preds = [], [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                avg_probs = tta_predict(model, inputs, device, n_aug=n_aug)
                all_probs.extend(avg_probs[:, 1].cpu().tolist())
                all_labels.extend(labels.tolist())
                all_preds.extend(avg_probs.argmax(1).cpu().tolist())

        results = {
            'auc': roc_auc_score(all_labels, all_probs),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'probs': all_probs,
            'labels': all_labels,
            'preds': all_preds,
        }
        breakdown = per_source_breakdown(test_dataset, all_preds, all_labels)

        # Measure TTA latency
        dummy = torch.randn(1, 1, 64, 188).to(device)
        for _ in range(10):
            with torch.no_grad():
                tta_predict(model, dummy, device, n_aug=n_aug)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                tta_predict(model, dummy, device, n_aug=n_aug)
            times.append((time.perf_counter() - start) * 1000)
        lat_mean, lat_std = np.mean(times), np.std(times)

        params = count_parameters(model)
        size = model_size_mb(model)

        log_experiment(
            f"TTA ({n_aug} augmentations)", results, breakdown,
            lat_mean, lat_std, params, size,
            f"{n_aug}x: orig+freq±2+time±10"
        )

        print(f"  AUC={results['auc']:.4f}, Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}")
        print(f"  Latency: {lat_mean:.1f}±{lat_std:.1f}ms")
        for src, data in sorted(breakdown.items()):
            print(f"    {src}: {data['accuracy']:.3f} ({data['correct']}/{data['total']})")


if __name__ == "__main__":
    main()
