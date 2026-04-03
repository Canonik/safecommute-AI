"""
Experiment 3: Model Ensemble.
Average softmax outputs from multiple trained checkpoints.
Uses existing checkpoints + trains variants with different hyperparameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH
from research.experiments.eval_utils import (
    get_test_loader, per_source_breakdown,
    count_parameters, model_size_mb, log_experiment
)


def train_variant(gamma, save_path):
    """Train a variant with different gamma for diversity."""
    from v_3.train_experimental import train
    print(f"\n  Training variant with gamma={gamma}...")
    train(use_focal=True, use_cosine=True, use_strong_aug=True,
          gamma=gamma, save_path=save_path)


def ensemble_predict(models, inputs, device):
    """Average softmax across all models."""
    all_probs = []
    for model in models:
        logits = model(inputs.to(device))
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs)
    return torch.stack(all_probs).mean(dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("research/results", exist_ok=True)

    # Available checkpoints
    checkpoint_paths = []
    for path in [MODEL_SAVE_PATH, "best_clean_g3.pth"]:
        if os.path.exists(path):
            checkpoint_paths.append(path)
            print(f"  Found checkpoint: {path}")

    # Train 2 additional variants for diversity
    variants = [
        (2.0, "research/results/ensemble_g2.pth"),
        (5.0, "research/results/ensemble_g5.pth"),
    ]

    for gamma, save_path in variants:
        if not os.path.exists(save_path):
            train_variant(gamma, save_path)
        checkpoint_paths.append(save_path)

    print(f"\n=== Ensemble with {len(checkpoint_paths)} models ===")

    # Load all models
    models = []
    for path in checkpoint_paths:
        m = SafeCommuteCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
        print(f"  Loaded: {path}")

    # Evaluate ensemble
    test_dataset, test_loader = get_test_loader()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            avg_probs = ensemble_predict(models, inputs, device)
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

    # Measure ensemble latency
    dummy = torch.randn(1, 1, 64, 188).to(device)
    for _ in range(10):
        with torch.no_grad():
            ensemble_predict(models, dummy, device)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            ensemble_predict(models, dummy, device)
        times.append((time.perf_counter() - start) * 1000)
    lat_mean, lat_std = np.mean(times), np.std(times)

    total_params = sum(count_parameters(m) for m in models)
    total_size = sum(model_size_mb(m) for m in models)

    log_experiment(
        f"Ensemble ({len(models)} models)", results, breakdown,
        lat_mean, lat_std, total_params, total_size,
        f"γ variants: {[p.split('/')[-1] for p in checkpoint_paths]}"
    )

    print(f"  AUC={results['auc']:.4f}, Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}")
    print(f"  Latency: {lat_mean:.1f}±{lat_std:.1f}ms")
    for src, data in sorted(breakdown.items()):
        print(f"    {src}: {data['accuracy']:.3f} ({data['correct']}/{data['total']})")


if __name__ == "__main__":
    main()
