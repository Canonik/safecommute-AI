"""
Leave-One-Source-Out (LOSO) Evaluation (v2).

For each unique data source, train on everything EXCEPT that source,
then evaluate on the held-out source only. This is the ultimate
generalization test: can the model detect threats from audio sources
it has never seen during training?

v2 changes:
  - Source extraction handles v2 naming (as_screaming, yt_metro, etc.)
  - Uses dataset-level augmentation (augment=True) instead of deleted spec_augment_strong
  - Batch-level GPU strong augmentation matching train.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from datetime import datetime

from safecommute.model import SafeCommuteCNN
from safecommute.constants import DATA_DIR, STATS_PATH
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything, worker_init_fn
from safecommute.pipeline.train import FocalLoss, mixup_batch

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
MIXUP_PROB = 0.5


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


def get_source_from_filename(fname):
    """
    Extract meaningful source prefix from .pt filename.

    v2 naming conventions:
      as_screaming_{id}_c000.pt  → as_screaming
      as_crowd_{id}_c001.pt      → as_crowd
      yt_metro_{id}_c000.pt      → yt_metro
      yt_scream_{id}_c000.pt     → yt_scream
      viol_violence_{id}_c000.pt → viol
      bg_{id}.pt                 → bg
      hns_{id}.pt                → hns
      esc_{name}.pt              → esc
      esc_hns_{name}.pt          → esc
    """
    parts = fname.split('_')
    if parts[0] == 'as' and len(parts) > 1:
        return f"as_{parts[1]}"  # as_screaming, as_crowd, etc.
    elif parts[0] == 'yt' and len(parts) > 1:
        return f"yt_{parts[1]}"  # yt_metro, yt_scream
    elif parts[0] == 'viol':
        return 'viol'
    elif parts[0] == 'esc':
        return 'esc'
    elif parts[0] == 'fsd' and len(parts) > 1:
        return f"fsd_{parts[1]}"
    else:
        return parts[0]  # bg, hns


def get_source_indices(dataset):
    """Map each source to its sample indices."""
    source_map = defaultdict(list)
    for i, path in enumerate(dataset.filepaths):
        fname = os.path.basename(path)
        source = get_source_from_filename(fname)
        source_map[source].append(i)
    return dict(source_map)


def train_and_evaluate(train_indices, test_indices, dataset, device):
    """Train on subset, evaluate on held-out source."""
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SafeCommuteCNN().to(device)

    train_labels = [dataset.labels[i] for i in train_indices]
    counts = [train_labels.count(0), train_labels.count(1)]
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    class_wts = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0
    best_state = None

    # Use 10% of training data as validation for early stopping
    val_size = max(1, len(train_indices) // 10)
    perm = np.random.permutation(len(train_indices))
    inner_val_indices = [train_indices[i] for i in perm[:val_size]]
    inner_train_indices = [train_indices[i] for i in perm[val_size:]]

    inner_train_loader = DataLoader(Subset(dataset, inner_train_indices),
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=2, pin_memory=True,
                                    worker_init_fn=worker_init_fn)
    inner_val_loader = DataLoader(Subset(dataset, inner_val_indices),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in inner_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Per-sample GPU augmentation (matching train.py)
            B = inputs.size(0)
            noise_mask = torch.rand(B, 1, 1, 1, device=device) < 0.3
            inputs = inputs + noise_mask * torch.randn_like(inputs) * 0.1
            for i in range(B):
                if random.random() < 0.3:
                    shift = random.randint(-20, 20)
                    inputs[i] = torch.roll(inputs[i], shifts=shift, dims=-1)
                if random.random() < 0.2:
                    f_start = random.randint(0, 50)
                    f_width = random.randint(3, 10)
                    inputs[i, :, f_start:f_start + f_width, :] = 0

            if np.random.random() < MIXUP_PROB:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, MIXUP_ALPHA)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step(epoch)

        # Inner validation for early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_inp, v_lab in inner_val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                val_loss += criterion(model(v_inp), v_lab).item()

        val_loss_avg = val_loss / max(len(inner_val_loader), 1)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_impro = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= PATIENCE:
                break

    # Evaluate on held-out source
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())

    # AUC requires both classes present
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return auc, acc, f1


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    print("=== Leave-One-Source-Out (LOSO) Evaluation v2 ===\n")

    # Load ALL data merged (with augmentation for training subsets)
    all_filepaths = []
    all_labels_list = []
    for split in ['train', 'val', 'test']:
        ds = TensorAudioDataset(os.path.join(DATA_DIR, split), mean, std, augment=True)
        all_filepaths.extend(ds.filepaths)
        all_labels_list.extend(ds.labels)

    merged = TensorAudioDataset.__new__(TensorAudioDataset)
    merged.filepaths = all_filepaths
    merged.labels = all_labels_list
    merged.mean = mean
    merged.std = std
    merged.augment = True  # augmentation for training subsets
    merged.load_teacher = False
    merged.freq_mask = ds.freq_mask
    merged.time_mask = ds.time_mask

    source_indices = get_source_indices(merged)
    sources = sorted(source_indices.keys())

    print(f"  Total: {len(merged)} samples across {len(sources)} sources")
    for src in sources:
        n = len(source_indices[src])
        n_unsafe = sum(1 for i in source_indices[src] if merged.labels[i] == 1)
        print(f"    {src}: {n} samples ({n_unsafe} unsafe)")

    results = []
    for held_out in sources:
        print(f"\n--- Held out: {held_out} ({len(source_indices[held_out])} samples) ---")

        test_indices = source_indices[held_out]
        train_indices = []
        for src in sources:
            if src != held_out:
                train_indices.extend(source_indices[src])

        print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

        auc, acc, f1 = train_and_evaluate(train_indices, test_indices, merged, device)
        results.append({'source': held_out, 'auc': auc, 'accuracy': acc, 'f1': f1,
                        'n_samples': len(test_indices)})
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"  {held_out}: AUC={auc_str}, Acc={acc:.4f}, F1={f1:.4f}")

    # Summary
    valid = [r for r in results if not np.isnan(r['auc'])]
    aucs = [r['auc'] for r in valid]

    print(f"\n{'='*60}")
    print(f"  LOSO v2 Results")
    print(f"{'='*60}")
    print(f"\n  | Source | AUC | Accuracy | F1 | Samples |")
    print(f"  |--------|-----|----------|----|---------| ")
    for r in sorted(results, key=lambda x: x.get('auc', 0) if not np.isnan(x.get('auc', 0)) else -1, reverse=True):
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "N/A (single class)"
        print(f"  | {r['source']} | {auc_str} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['n_samples']} |")

    if aucs:
        print(f"\n  Mean AUC (sources with both classes): {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        best = max(valid, key=lambda x: x['auc'])
        worst = min(valid, key=lambda x: x['auc'])
        print(f"  Easiest to generalize: {best['source']} (AUC={best['auc']:.4f})")
        print(f"  Hardest to generalize: {worst['source']} (AUC={worst['auc']:.4f})")

    # Save
    os.makedirs("research/results", exist_ok=True)
    save_data = {
        'method': 'Leave-One-Source-Out (LOSO) v2',
        'results': results,
        'mean_auc': float(np.mean(aucs)) if aucs else None,
        'std_auc': float(np.std(aucs)) if aucs else None,
        'timestamp': datetime.now().isoformat(),
    }
    with open("research/results/loso_v2_results.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # Append to experiment log
    with open("research/experiment_log.md", 'a') as f:
        f.write(f"\n## Leave-One-Source-Out (LOSO) v2\n\n")
        f.write(f"| Held-Out Source | AUC | Accuracy | F1 | Samples |\n")
        f.write(f"|-----------------|-----|----------|----|---------|\n")
        for r in sorted(results, key=lambda x: x.get('auc', 0) if not np.isnan(x.get('auc', 0)) else -1, reverse=True):
            auc_str = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "N/A"
            f.write(f"| {r['source']} | {auc_str} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['n_samples']} |\n")
        if aucs:
            f.write(f"| **Mean** | **{np.mean(aucs):.4f}** | | | |\n")
            f.write(f"| **Std** | **{np.std(aucs):.4f}** | | | |\n")
        f.write(f"\nRun: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

    print(f"\n  Results saved to research/results/loso_v2_results.json")


if __name__ == "__main__":
    main()
