"""
Temperature scaling for SafeCommute AI.

Learns a single temperature parameter T on the validation set that
sharpens or softens probability outputs without changing ranking (AUC).

calibrated_probs = softmax(logits / T)
  T < 1: sharper (more confident)
  T > 1: softer (less confident)

Usage:
    PYTHONPATH=. python safecommute/pipeline/calibrate.py
    PYTHONPATH=. python safecommute/pipeline/calibrate.py --model models/metro_model.pth
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH


def find_optimal_temperature(model, val_loader, device):
    """Find T that minimizes NLL on validation set."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    best_t, best_nll = 1.0, float('inf')
    for t in np.arange(0.1, 5.01, 0.05):
        scaled = all_logits / t
        nll = F.cross_entropy(scaled, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return best_t, best_nll


def evaluate_calibration(model, loader, device, temperature=1.0):
    """Compute ECE and probability distribution stats."""
    model.eval()
    safe_probs, unsafe_probs = [], []
    all_probs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device)) / temperature
            probs = torch.softmax(logits, dim=1)[:, 1]
            for p, l in zip(probs.cpu().tolist(), labels.tolist()):
                all_probs.append(p)
                all_labels.append(l)
                (unsafe_probs if l == 1 else safe_probs).append(p)

    # ECE (Expected Calibration Error)
    n_bins = 10
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = [(lo <= p < hi) for p in all_probs]
        if not any(mask):
            continue
        bin_probs = [p for p, m in zip(all_probs, mask) if m]
        bin_labels = [l for l, m in zip(all_labels, mask) if m]
        bin_acc = np.mean(bin_labels)
        bin_conf = np.mean(bin_probs)
        ece += len(bin_probs) / len(all_probs) * abs(bin_acc - bin_conf)

    return {
        'safe_mean': float(np.mean(safe_probs)),
        'safe_median': float(np.median(safe_probs)),
        'unsafe_mean': float(np.mean(unsafe_probs)),
        'unsafe_median': float(np.median(unsafe_probs)),
        'separation': float(np.mean(unsafe_probs) - np.mean(safe_probs)),
        'ece': float(ece),
        'safe_correct_at_05': float(sum(1 for p in safe_probs if p < 0.5) / max(len(safe_probs), 1)),
        'unsafe_correct_at_05': float(sum(1 for p in unsafe_probs if p > 0.5) / max(len(unsafe_probs), 1)),
    }


def main():
    parser = argparse.ArgumentParser(description='Temperature scaling calibration')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH)
    args = parser.parse_args()

    device = torch.device('cpu')

    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    val_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    test_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    print("=" * 60)
    print(" Temperature Scaling Calibration")
    print("=" * 60)

    # Before calibration
    print("\n  Before calibration (T=1.0):")
    before = evaluate_calibration(model, test_loader, device, temperature=1.0)
    print(f"    Safe mean:   {before['safe_mean']:.3f}  Unsafe mean: {before['unsafe_mean']:.3f}")
    print(f"    Separation:  {before['separation']:.3f}  ECE: {before['ece']:.3f}")
    print(f"    Safe@0.5:    {before['safe_correct_at_05']:.1%}  Unsafe@0.5: {before['unsafe_correct_at_05']:.1%}")

    # Find optimal T on val set
    print("\n  Optimizing temperature on validation set...")
    best_t, best_nll = find_optimal_temperature(model, val_loader, device)
    print(f"    Optimal T = {best_t:.2f} (NLL = {best_nll:.4f})")

    # After calibration
    print(f"\n  After calibration (T={best_t:.2f}):")
    after = evaluate_calibration(model, test_loader, device, temperature=best_t)
    print(f"    Safe mean:   {after['safe_mean']:.3f}  Unsafe mean: {after['unsafe_mean']:.3f}")
    print(f"    Separation:  {after['separation']:.3f}  ECE: {after['ece']:.3f}")
    print(f"    Safe@0.5:    {after['safe_correct_at_05']:.1%}  Unsafe@0.5: {after['unsafe_correct_at_05']:.1%}")

    # Silence test
    from safecommute.features import preprocess
    silence = np.zeros(48000, dtype=np.float32)
    feat = preprocess(silence, mean, std)
    with torch.no_grad():
        logits = model(feat)
        p_before = torch.softmax(logits, dim=1)[0][1].item()
        p_after = torch.softmax(logits / best_t, dim=1)[0][1].item()
    print(f"\n  Silence unsafe prob: {p_before:.3f} → {p_after:.3f} (T={best_t:.2f})")

    # Save
    temp_path = 'models/temperature.json'
    os.makedirs('models', exist_ok=True)
    with open(temp_path, 'w') as f:
        json.dump({'temperature': best_t, 'nll': best_nll,
                   'before': before, 'after': after}, f, indent=2)
    print(f"\n  Saved: {temp_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  {'':18} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'Safe mean':<18} {before['safe_mean']:>10.3f} {after['safe_mean']:>10.3f} {after['safe_mean']-before['safe_mean']:>+10.3f}")
    print(f"  {'Unsafe mean':<18} {before['unsafe_mean']:>10.3f} {after['unsafe_mean']:>10.3f} {after['unsafe_mean']-before['unsafe_mean']:>+10.3f}")
    print(f"  {'Separation':<18} {before['separation']:>10.3f} {after['separation']:>10.3f} {after['separation']-before['separation']:>+10.3f}")
    print(f"  {'ECE':<18} {before['ece']:>10.3f} {after['ece']:>10.3f} {after['ece']-before['ece']:>+10.3f}")


if __name__ == "__main__":
    main()
