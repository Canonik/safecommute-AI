"""
5-Fold Cross-Validation with source-aware stratification.
Ensures no source leakage: segments from the same video stay in the same fold.
Reports mean +/- std AUC, accuracy, F1 across folds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from safecommute.model import SafeCommuteCNN
from safecommute.constants import DATA_DIR, STATS_PATH
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from safecommute.pipeline.train import FocalLoss, mixup_batch, compute_class_weights
import random

K = 5
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


def get_source_groups(dataset):
    """Group samples by source (prefix before first underscore).
    Segments from the same video (same prefix+id) stay together."""
    groups = defaultdict(list)
    for i, path in enumerate(dataset.filepaths):
        fname = os.path.basename(path)
        # Group by source + video ID to prevent leakage
        # e.g., "yt_metro_abc123_c001.pt" -> "yt_metro_abc123"
        parts = fname.rsplit('_c', 1)  # split off clip number
        group_key = parts[0] if len(parts) > 1 else fname.replace('.pt', '')
        groups[group_key].append(i)
    return groups


def train_fold(train_indices, val_indices, full_dataset, device, fold_num):
    """Train one fold, return metrics on held-out validation."""
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SafeCommuteCNN().to(device)

    # Compute class weights from training fold
    train_labels = [full_dataset.labels[i] for i in train_indices]
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

    for epoch in range(EPOCHS):
        model.train()
        run_loss = correct = total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            B = inputs.size(0)
            noise_mask = torch.rand(B, 1, 1, 1, device=inputs.device) < 0.3
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
                c_lab = labels_a if lam >= 0.5 else labels_b
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                c_lab = labels

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = outputs.detach().argmax(1)
            run_loss += loss.item()
            correct += (preds == c_lab).sum().item()
            total_samples += labels.size(0)

        scheduler.step(epoch)

        # Validation
        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for v_inp, v_lab in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                v_out = model(v_inp)
                val_loss += criterion(v_out, v_lab).item()
                v_correct += (v_out.argmax(1) == v_lab).sum().item()
                v_total += v_lab.size(0)

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100 * v_correct / v_total
        train_acc = 100 * correct / total_samples

        print(f"    E{epoch+1:>2} TA={train_acc:.1f}% VA={val_acc:.1f}%", end='')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_impro = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(" *")
        else:
            epochs_no_impro += 1
            print()
            if epochs_no_impro >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

    # Evaluate best model on val fold
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return auc, acc, f1


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    print("=== 5-Fold Cross-Validation (source-aware) ===\n")

    # Load ALL data (train + val + test combined)
    all_datasets = []
    for split in ['train', 'val', 'test']:
        ds = TensorAudioDataset(os.path.join(DATA_DIR, split), mean, std)
        all_datasets.append(ds)

    # Merge into one dataset
    all_filepaths = []
    all_labels_list = []
    for ds in all_datasets:
        all_filepaths.extend(ds.filepaths)
        all_labels_list.extend(ds.labels)

    # Create a merged dataset
    merged = TensorAudioDataset.__new__(TensorAudioDataset)
    merged.filepaths = all_filepaths
    merged.labels = all_labels_list
    merged.mean = mean
    merged.std = std
    merged.load_teacher = False

    print(f"  Total samples: {len(merged)} ({sum(1 for l in merged.labels if l==0)} safe, "
          f"{sum(1 for l in merged.labels if l==1)} unsafe)")

    # Get source-aware groups
    groups = get_source_groups(merged)
    print(f"  Source groups: {len(groups)}")

    # Create group-level labels for stratification
    group_keys = list(groups.keys())
    group_labels = []
    for key in group_keys:
        indices = groups[key]
        # Use majority label for the group
        labels = [merged.labels[i] for i in indices]
        group_labels.append(1 if sum(labels) > len(labels) / 2 else 0)

    group_labels = np.array(group_labels)

    # Stratified K-Fold on groups
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_group_idx, val_group_idx) in enumerate(skf.split(group_keys, group_labels)):
        print(f"\n--- Fold {fold+1}/{K} ---")

        # Map group indices back to sample indices
        train_indices = []
        for gi in train_group_idx:
            train_indices.extend(groups[group_keys[gi]])
        val_indices = []
        for gi in val_group_idx:
            val_indices.extend(groups[group_keys[gi]])

        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

        auc, acc, f1 = train_fold(train_indices, val_indices, merged, device, fold + 1)
        fold_results.append({'auc': auc, 'accuracy': acc, 'f1': f1})
        print(f"  Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Summary
    aucs = [r['auc'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    print(f"\n{'='*50}")
    print(f"  5-Fold Cross-Validation Results")
    print(f"{'='*50}")
    print(f"  AUC:      {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  F1:       {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"\n  Per-fold: {[f'{a:.4f}' for a in aucs]}")

    # Save results
    results = {
        'method': '5-fold cross-validation (source-aware stratified)',
        'folds': fold_results,
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs("research/results", exist_ok=True)
    with open("research/results/cross_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Append to experiment log
    with open("research/experiment_log.md", 'a') as f:
        f.write(f"\n## 5-Fold Cross-Validation (source-aware)\n\n")
        f.write(f"| Fold | AUC | Accuracy | F1 |\n")
        f.write(f"|------|-----|----------|----|\n")
        for i, r in enumerate(fold_results):
            f.write(f"| {i+1} | {r['auc']:.4f} | {r['accuracy']:.4f} | {r['f1']:.4f} |\n")
        f.write(f"| **Mean** | **{np.mean(aucs):.4f}** | **{np.mean(accs):.4f}** | **{np.mean(f1s):.4f}** |\n")
        f.write(f"| **Std** | **{np.std(aucs):.4f}** | **{np.std(accs):.4f}** | **{np.std(f1s):.4f}** |\n")
        f.write(f"\nRun: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

    print(f"\n  Results saved to research/results/cross_validation_results.json")


if __name__ == "__main__":
    main()
