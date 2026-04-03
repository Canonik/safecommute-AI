"""
Shared evaluation utilities for all research experiments.
Provides consistent metrics across all experiments.
"""
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from datetime import datetime

from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


def get_test_loader(batch_size=32):
    mean, std = load_stats()
    test_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_dataset, test_loader


def get_train_val_loaders(batch_size=32, load_teacher=False):
    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std, load_teacher=load_teacher)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dataset, val_dataset, train_loader, val_loader


def evaluate_model(model, test_loader, device):
    """Evaluate model and return dict with AUC, accuracy, F1, per-source breakdown."""
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    all_paths = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0], batch[1]
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
    }


def per_source_breakdown(test_dataset, all_preds, all_labels):
    """Compute per-source accuracy from filename prefixes."""
    source_results = {}
    for i, path in enumerate(test_dataset.filepaths):
        fname = os.path.basename(path)
        # Extract source prefix (e.g., 'cremad', 'yt', 'bg', 'esc', 'tess', 'rav', 'hns', 'viol', 'savee')
        source = fname.split('_')[0]
        if source not in source_results:
            source_results[source] = {'correct': 0, 'total': 0}
        source_results[source]['total'] += 1
        if all_preds[i] == all_labels[i]:
            source_results[source]['correct'] += 1

    breakdown = {}
    for src, data in sorted(source_results.items()):
        breakdown[src] = {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'total': data['total'],
            'correct': data['correct'],
        }
    return breakdown


def measure_latency(model, device, n_runs=100):
    """Measure average inference latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 1, 64, 188).to(device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """Approximate model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def log_experiment(name, results, source_breakdown, latency_ms, latency_std, params, size_mb,
                   notes="", log_path="research/experiment_log.md"):
    """Append experiment result to the log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(log_path):
        header = """# Experiment Log

| # | Experiment | AUC | Accuracy | F1 | Params | Size(MB) | Latency(ms) | Notes |
|---|-----------|-----|----------|----|---------|---------|-----------:|-------|
"""
        with open(log_path, 'w') as f:
            f.write(header)

    # Count existing experiments
    with open(log_path) as f:
        lines = f.readlines()
    num = sum(1 for l in lines if l.startswith('|') and not l.startswith('| #') and not l.startswith('|---'))

    entry = (f"| {num+1} | {name} | {results['auc']:.4f} | {results['accuracy']:.4f} | "
             f"{results['f1']:.4f} | {params:,} | {size_mb:.2f} | {latency_ms:.1f}±{latency_std:.1f} | {notes} |\n")

    with open(log_path, 'a') as f:
        f.write(entry)
        f.write(f"\n### {name} — Per-Source Breakdown\n")
        f.write("| Source | Accuracy | Correct/Total |\n|--------|----------|---------------|\n")
        for src, data in sorted(source_breakdown.items()):
            f.write(f"| {src} | {data['accuracy']:.3f} | {data['correct']}/{data['total']} |\n")
        f.write(f"\nRun: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

    print(f"\n✓ Logged: {name} — AUC={results['auc']:.4f}, Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}")


def full_evaluation(model, device, experiment_name, notes=""):
    """Run complete evaluation pipeline: metrics, per-source, latency, log."""
    test_dataset, test_loader = get_test_loader()
    results = evaluate_model(model, test_loader, device)
    breakdown = per_source_breakdown(test_dataset, results['preds'], results['labels'])
    lat_mean, lat_std = measure_latency(model, device)
    params = count_parameters(model)
    size = model_size_mb(model)

    log_experiment(experiment_name, results, breakdown, lat_mean, lat_std, params, size, notes)

    print(f"  Per-source:")
    for src, data in sorted(breakdown.items()):
        print(f"    {src}: {data['accuracy']:.3f} ({data['correct']}/{data['total']})")

    return results, breakdown
