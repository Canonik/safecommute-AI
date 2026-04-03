"""
Ablation Study: prove each architectural component contributes.
Systematically remove one component at a time and measure the drop.

Variants:
  1. Full model (baseline)
  2. No SE blocks (remove channel attention)
  3. No GRU (replace with global average pooling)
  4. No multi-scale pooling (use only last hidden state)
  5. Half channels (64→32, 128→64, 256→128)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from datetime import datetime

from safecommute.model import SafeCommuteCNN, SEBlock, ConvBlock
from safecommute.constants import DATA_DIR, STATS_PATH, N_MELS
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from safecommute.pipeline.train import FocalLoss, spec_augment_strong, mixup_batch
from research.experiments.eval_utils import load_stats, count_parameters, model_size_mb

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6


# ---- Variant 2: No SE ----
class ConvBlock_NoSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, pool=(2, 2)):
        return F.avg_pool2d(self.conv(x), pool)

class SafeCommuteCNN_NoSE(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock_NoSE(1, 64)
        self.block2 = ConvBlock_NoSE(64, 128)
        self.block3 = ConvBlock_NoSE(128, 256)
        self.freq_reduce = nn.Linear(256 * (n_mels // 8), 256)
        self.gru = nn.GRU(256, 128, 1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(384, n_classes)
    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))
        out, h = self.gru(x)
        x = torch.cat([h.squeeze(0), out.mean(1), out.max(1).values], dim=1)
        return self.fc(self.dropout(x))


# ---- Variant 3: No GRU (global pooling instead) ----
class SafeCommuteCNN_NoGRU(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(self.dropout(x))


# ---- Variant 4: No multi-scale pooling (last hidden only) ----
class SafeCommuteCNN_LastOnly(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.freq_reduce = nn.Linear(256 * (n_mels // 8), 256)
        self.gru = nn.GRU(256, 128, 1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)  # only last hidden
    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))
        _, h = self.gru(x)
        return self.fc(self.dropout(h.squeeze(0)))


# ---- Variant 5: Half channels ----
class SafeCommuteCNN_Half(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.freq_reduce = nn.Linear(128 * (n_mels // 8), 128)
        self.gru = nn.GRU(128, 64, 1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(192, n_classes)
    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))
        out, h = self.gru(x)
        x = torch.cat([h.squeeze(0), out.mean(1), out.max(1).values], dim=1)
        return self.fc(self.dropout(x))


def train_and_eval(model, name, device):
    """Train a model and evaluate on test set."""
    mean, std = load_stats()
    train_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    test_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    counts = [train_ds.labels.count(0), train_ds.labels.count(1)]
    total = sum(counts)
    weights = [total / (2 * c) for c in counts]
    class_wts = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            aug = [spec_augment_strong(inputs[i].cpu()).to(device) for i in range(inputs.size(0))]
            inputs = torch.stack(aug)

            if np.random.random() < 0.5:
                inputs, la, lb, lam = mixup_batch(inputs, labels, 0.3)
                loss = lam * criterion(model(inputs), la) + (1-lam) * criterion(model(inputs), lb)
            else:
                loss = criterion(model(inputs), labels)

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step(epoch)

        model.eval()
        vl = 0
        with torch.no_grad():
            for vi, vl_ in val_loader:
                vl += criterion(model(vi.to(device)), vl_.to(device)).item()
        vl /= max(len(val_loader), 1)

        if vl < best_val_loss:
            best_val_loss = vl; epochs_no_impro = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs_all.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            labels_all.extend(labels.tolist())

    auc = roc_auc_score(labels_all, probs_all)
    acc = accuracy_score(labels_all, [1 if p > 0.5 else 0 for p in probs_all])
    f1 = f1_score(labels_all, [1 if p > 0.5 else 0 for p in probs_all], average='weighted')
    params = count_parameters(model)

    return {'name': name, 'auc': auc, 'accuracy': acc, 'f1': f1, 'params': params}


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = [
        ("Full model (baseline)", SafeCommuteCNN().to(device)),
        ("No SE attention", SafeCommuteCNN_NoSE().to(device)),
        ("No GRU (global pool)", SafeCommuteCNN_NoGRU().to(device)),
        ("No multi-scale pool", SafeCommuteCNN_LastOnly().to(device)),
        ("Half channels", SafeCommuteCNN_Half().to(device)),
    ]

    print("=== Ablation Study ===\n")
    results = []

    for name, model in variants:
        print(f"\n--- {name} ({count_parameters(model):,} params) ---")
        r = train_and_eval(model, name, device)
        results.append(r)
        print(f"  AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")

    # Summary
    baseline_auc = results[0]['auc']
    print(f"\n{'='*60}")
    print(f"  Ablation Study Results")
    print(f"{'='*60}")
    print(f"  | Variant | AUC | Acc | Params | Delta AUC |")
    print(f"  |---------|-----|-----|--------|-----------|")
    for r in results:
        delta = r['auc'] - baseline_auc
        print(f"  | {r['name']} | {r['auc']:.4f} | {r['accuracy']:.4f} | "
              f"{r['params']:,} | {delta:+.4f} |")

    # Save
    os.makedirs("research/results", exist_ok=True)
    with open("research/results/ablation_results.json", 'w') as f:
        json.dump({'results': results, 'timestamp': datetime.now().isoformat()}, f, indent=2)

    with open("research/experiment_log.md", 'a') as f:
        f.write(f"\n## Ablation Study\n\n")
        f.write(f"| Variant | AUC | Accuracy | F1 | Params | vs Full |\n")
        f.write(f"|---------|-----|----------|----|---------|---------|\n")
        for r in results:
            delta = r['auc'] - baseline_auc
            f.write(f"| {r['name']} | {r['auc']:.4f} | {r['accuracy']:.4f} | "
                    f"{r['f1']:.4f} | {r['params']:,} | {delta:+.4f} |\n")
        f.write(f"\nRun: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")


if __name__ == "__main__":
    main()
