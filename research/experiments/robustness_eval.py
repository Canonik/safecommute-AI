"""
Robustness Evaluation for SafeCommute AI.
Tests whether the model generalizes vs overfits.
Key metrics: train-test gap, per-source variance, noise robustness,
confidence calibration, decision boundary analysis.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
from collections import defaultdict
from datetime import datetime

from safecommute.model import SafeCommuteCNN
from safecommute.constants import DATA_DIR, MODEL_SAVE_PATH
from safecommute.dataset import TensorAudioDataset
from research.experiments.eval_utils import load_stats


def evaluate_on_split(model, dataset, device):
    """Get predictions on a split."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    all_probs, all_labels, all_preds = [], [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return auc, acc, f1, all_probs, all_labels, all_preds


def noise_robustness_test(model, dataset, device, noise_levels=[0.05, 0.1, 0.2, 0.5, 1.0]):
    """Test model performance under Gaussian noise corruption."""
    results = {}
    for noise_std in noise_levels:
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        all_probs, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                # Add Gaussian noise
                noisy = inputs + torch.randn_like(inputs) * noise_std
                logits = model(noisy.to(device))
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.tolist())

        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        results[noise_std] = {'auc': auc, 'accuracy': acc}

    return results


def time_masking_robustness(model, dataset, device, mask_ratios=[0.1, 0.2, 0.3, 0.5]):
    """Test robustness to time masking (simulating partial audio)."""
    results = {}
    for ratio in mask_ratios:
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        all_probs, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                B, C, H, W = inputs.shape
                mask_width = int(W * ratio)
                start = np.random.randint(0, W - mask_width)
                masked = inputs.clone()
                masked[:, :, :, start:start+mask_width] = 0
                logits = model(masked.to(device))
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.tolist())

        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        results[ratio] = {'auc': auc, 'accuracy': acc}

    return results


def confidence_calibration(probs, labels):
    """Measure confidence calibration (ECE, Brier score)."""
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)

    brier = brier_score_loss(labels_arr, probs_arr)

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (probs_arr >= bin_boundaries[i]) & (probs_arr < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = labels_arr[mask].mean()
            bin_conf = probs_arr[mask].mean()
            ece += mask.sum() / len(probs_arr) * abs(bin_acc - bin_conf)

    return {'brier_score': float(brier), 'ece': float(ece)}


def per_source_generalization(model, test_dataset, device):
    """Measure variance across sources — low variance = good generalization."""
    source_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    idx = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            preds = logits.argmax(1).cpu()
            for i in range(labels.size(0)):
                fname = os.path.basename(test_dataset.filepaths[idx])
                source = fname.split('_')[0]
                source_acc[source]['total'] += 1
                if preds[i] == labels[i]:
                    source_acc[source]['correct'] += 1
                idx += 1

    accuracies = {}
    for src, data in source_acc.items():
        accuracies[src] = data['correct'] / data['total']

    values = list(accuracies.values())
    return {
        'per_source': accuracies,
        'mean_accuracy': float(np.mean(values)),
        'std_accuracy': float(np.std(values)),
        'min_accuracy': float(np.min(values)),
        'max_accuracy': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
    }


def generate_robustness_report(results, save_path="research/robustness_report.md"):
    """Generate comprehensive robustness report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("# Robustness Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Goal:** Verify the model generalizes well and doesn't just overfit training data.\n\n")

        # 1. Train-Test Gap
        f.write("## 1. Train-Test Gap (Overfitting Indicator)\n\n")
        f.write("| Metric | Train | Val | Test | Train-Test Gap |\n")
        f.write("|--------|-------|-----|------|----------------|\n")
        for metric in ['auc', 'accuracy', 'f1']:
            train_val = results['splits']['train'][metric]
            val_val = results['splits']['val'][metric]
            test_val = results['splits']['test'][metric]
            gap = train_val - test_val
            status = "OK" if abs(gap) < 0.05 else "WARN" if abs(gap) < 0.1 else "OVERFIT"
            f.write(f"| {metric.upper()} | {train_val:.4f} | {val_val:.4f} | "
                    f"{test_val:.4f} | {gap:+.4f} ({status}) |\n")

        train_test_auc_gap = results['splits']['train']['auc'] - results['splits']['test']['auc']
        if train_test_auc_gap < 0.03:
            f.write("\n**Verdict: NO overfitting detected.** Train-test AUC gap is minimal.\n\n")
        elif train_test_auc_gap < 0.05:
            f.write("\n**Verdict: Slight overfitting.** Consider more regularization.\n\n")
        else:
            f.write(f"\n**Verdict: OVERFITTING DETECTED.** Gap of {train_test_auc_gap:.3f} is concerning.\n\n")

        # 2. Noise Robustness
        f.write("## 2. Noise Robustness\n\n")
        f.write("Adding Gaussian noise to spectrograms at test time:\n\n")
        f.write("| Noise σ | AUC | Accuracy | AUC Drop |\n")
        f.write("|---------|-----|----------|----------|\n")
        baseline_auc = results['splits']['test']['auc']
        for noise, data in sorted(results['noise'].items()):
            drop = baseline_auc - data['auc']
            f.write(f"| {noise:.2f} | {data['auc']:.4f} | {data['accuracy']:.4f} | {drop:+.4f} |\n")

        f.write("\n*A robust model should degrade gracefully with noise.*\n\n")

        # 3. Time Masking Robustness
        f.write("## 3. Time Masking Robustness\n\n")
        f.write("Masking portions of the spectrogram (simulating partial/corrupted audio):\n\n")
        f.write("| Mask Ratio | AUC | Accuracy | AUC Drop |\n")
        f.write("|------------|-----|----------|----------|\n")
        for ratio, data in sorted(results['time_masking'].items()):
            drop = baseline_auc - data['auc']
            f.write(f"| {ratio:.0%} | {data['auc']:.4f} | {data['accuracy']:.4f} | {drop:+.4f} |\n")

        f.write("\n*Moderate drops under masking are expected; large drops suggest fragile features.*\n\n")

        # 4. Confidence Calibration
        f.write("## 4. Confidence Calibration\n\n")
        cal = results['calibration']
        f.write(f"- **Brier Score:** {cal['brier_score']:.4f} (lower is better, <0.25 is good)\n")
        f.write(f"- **Expected Calibration Error (ECE):** {cal['ece']:.4f} (lower is better, <0.1 is good)\n\n")

        if cal['ece'] < 0.05:
            f.write("**Well calibrated.** Predicted probabilities match actual frequencies.\n\n")
        elif cal['ece'] < 0.1:
            f.write("**Reasonably calibrated.** Consider temperature scaling for deployment.\n\n")
        else:
            f.write("**Poorly calibrated.** Temperature scaling strongly recommended.\n\n")

        # 5. Cross-Source Generalization
        f.write("## 5. Cross-Source Generalization\n\n")
        gen = results['generalization']
        f.write(f"- **Mean accuracy across sources:** {gen['mean_accuracy']:.3f}\n")
        f.write(f"- **Std across sources:** {gen['std_accuracy']:.3f}\n")
        f.write(f"- **Range:** {gen['min_accuracy']:.3f} — {gen['max_accuracy']:.3f} "
                f"(spread = {gen['range']:.3f})\n\n")

        f.write("| Source | Accuracy | vs Mean |\n")
        f.write("|--------|----------|---------|\n")
        for src, acc in sorted(gen['per_source'].items(), key=lambda x: x[1]):
            delta = acc - gen['mean_accuracy']
            f.write(f"| {src} | {acc:.3f} | {delta:+.3f} |\n")

        if gen['std_accuracy'] < 0.15:
            f.write(f"\n**Good generalization.** Moderate variance across sources.\n\n")
        else:
            f.write(f"\n**High variance.** Model performs very differently across sources.\n\n")

        # 6. Deployment Readiness
        f.write("## 6. Deployment Readiness Summary\n\n")
        checks = [
            ("Overfitting", train_test_auc_gap < 0.05, f"AUC gap: {train_test_auc_gap:.3f}"),
            ("Noise robustness", results['noise'][0.2]['auc'] > 0.9, f"AUC@σ=0.2: {results['noise'][0.2]['auc']:.3f}"),
            ("Calibration", cal['ece'] < 0.1, f"ECE: {cal['ece']:.3f}"),
            ("Source generalization", gen['std_accuracy'] < 0.2, f"Std: {gen['std_accuracy']:.3f}"),
        ]

        f.write("| Check | Status | Detail |\n")
        f.write("|-------|--------|--------|\n")
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            f.write(f"| {name} | {status} | {detail} |\n")

        passed_count = sum(1 for _, p, _ in checks if p)
        f.write(f"\n**Score: {passed_count}/{len(checks)} checks passed.**\n")

    print(f"Report saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    model = SafeCommuteCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    results = {}

    # 1. Evaluate on all splits
    print("=== Robustness Evaluation ===")
    results['splits'] = {}
    for split in ['train', 'val', 'test']:
        print(f"  Evaluating on {split}...")
        dataset = TensorAudioDataset(os.path.join(DATA_DIR, split), mean, std)
        auc, acc, f1, probs, labels, preds = evaluate_on_split(model, dataset, device)
        results['splits'][split] = {'auc': auc, 'accuracy': acc, 'f1': f1}
        print(f"    AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    train_test_gap = results['splits']['train']['auc'] - results['splits']['test']['auc']
    print(f"\n  Train-Test AUC Gap: {train_test_gap:+.4f}")

    # 2. Noise robustness
    print("\n  Testing noise robustness...")
    test_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
    results['noise'] = noise_robustness_test(model, test_dataset, device)
    for noise, data in sorted(results['noise'].items()):
        print(f"    σ={noise:.2f}: AUC={data['auc']:.4f}")

    # 3. Time masking robustness
    print("\n  Testing time masking robustness...")
    results['time_masking'] = time_masking_robustness(model, test_dataset, device)
    for ratio, data in sorted(results['time_masking'].items()):
        print(f"    mask={ratio:.0%}: AUC={data['auc']:.4f}")

    # 4. Calibration
    print("\n  Computing calibration metrics...")
    _, _, _, probs, labels, _ = evaluate_on_split(model, test_dataset, device)
    results['calibration'] = confidence_calibration(probs, labels)
    print(f"    Brier={results['calibration']['brier_score']:.4f}, ECE={results['calibration']['ece']:.4f}")

    # 5. Cross-source generalization
    print("\n  Analyzing cross-source generalization...")
    results['generalization'] = per_source_generalization(model, test_dataset, device)
    print(f"    Mean acc: {results['generalization']['mean_accuracy']:.3f} "
          f"± {results['generalization']['std_accuracy']:.3f}")

    # Generate report
    generate_robustness_report(results)


if __name__ == "__main__":
    main()
