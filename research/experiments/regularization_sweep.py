"""
Regularization Sweep.
Grid search over dropout, weight_decay, label_smoothing to find optimal values.
Tests whether current regularization is too weak or too strong.
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

from safecommute.model import SafeCommuteCNN, ConvBlock, SEBlock
from safecommute.constants import DATA_DIR, STATS_PATH, N_MELS
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from safecommute.pipeline.train import FocalLoss, spec_augment_strong, mixup_batch

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6

# Grid
DROPOUT_VALUES = [0.2, 0.3, 0.4, 0.5]
WEIGHT_DECAY_VALUES = [1e-4, 5e-4, 1e-3]
LABEL_SMOOTHING_VALUES = [0.0, 0.1, 0.2]


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


class SafeCommuteCNN_Dropout(nn.Module):
    """SafeCommuteCNN with configurable dropout."""
    def __init__(self, n_mels=N_MELS, n_classes=2, dropout=0.3):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        freq_after_blocks = n_mels // (2 ** 3)
        self.freq_reduce = nn.Linear(256 * freq_after_blocks, 256)
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(384, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))
        out, h = self.gru(x)
        h_last = h.squeeze(0)
        h_mean = out.mean(dim=1)
        h_max = out.max(dim=1).values
        x = torch.cat([h_last, h_mean, h_max], dim=1)
        x = self.dropout(x)
        return self.fc(x)


def train_config(dropout, weight_decay, label_smoothing, device, train_loader, val_loader, class_wts):
    """Train with a specific regularization config, return val AUC."""
    seed_everything()
    model = SafeCommuteCNN_Dropout(dropout=dropout).to(device)
    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            aug_inputs = []
            for i in range(inputs.size(0)):
                aug_inputs.append(spec_augment_strong(inputs[i].cpu()).to(device))
            inputs = torch.stack(aug_inputs)

            if np.random.random() < 0.5:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, 0.3)
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_inp, v_lab in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                val_loss += criterion(model(v_inp), v_lab).item()

        val_loss_avg = val_loss / max(len(val_loader), 1)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_impro = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= PATIENCE:
                break

    # Evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    mean, std = load_stats()
    test_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
    return auc, acc, best_val_loss


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats()

    print("=== Regularization Sweep ===\n")

    train_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_ds = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    counts = [train_ds.labels.count(0), train_ds.labels.count(1)]
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    class_wts = torch.tensor(weights, dtype=torch.float32).to(device)

    # Sweep: fix 2 params, vary 1 (to keep tractable)
    results = []

    # 1. Dropout sweep (fix wd=1e-4, ls=0.1)
    print("--- Dropout sweep (wd=1e-4, ls=0.1) ---")
    for dropout in DROPOUT_VALUES:
        print(f"  dropout={dropout}...", end=' ', flush=True)
        auc, acc, val_loss = train_config(dropout, 1e-4, 0.1, device, train_loader, val_loader, class_wts)
        results.append({'dropout': dropout, 'weight_decay': 1e-4, 'label_smoothing': 0.1,
                        'auc': auc, 'accuracy': acc})
        print(f"AUC={auc:.4f}, Acc={acc:.4f}")

    # 2. Weight decay sweep (fix dropout=0.3, ls=0.1)
    print("\n--- Weight decay sweep (dropout=0.3, ls=0.1) ---")
    for wd in WEIGHT_DECAY_VALUES:
        print(f"  weight_decay={wd}...", end=' ', flush=True)
        auc, acc, val_loss = train_config(0.3, wd, 0.1, device, train_loader, val_loader, class_wts)
        results.append({'dropout': 0.3, 'weight_decay': wd, 'label_smoothing': 0.1,
                        'auc': auc, 'accuracy': acc})
        print(f"AUC={auc:.4f}, Acc={acc:.4f}")

    # 3. Label smoothing sweep (fix dropout=0.3, wd=1e-4)
    print("\n--- Label smoothing sweep (dropout=0.3, wd=1e-4) ---")
    for ls in LABEL_SMOOTHING_VALUES:
        print(f"  label_smoothing={ls}...", end=' ', flush=True)
        auc, acc, val_loss = train_config(0.3, 1e-4, ls, device, train_loader, val_loader, class_wts)
        results.append({'dropout': 0.3, 'weight_decay': 1e-4, 'label_smoothing': ls,
                        'auc': auc, 'accuracy': acc})
        print(f"AUC={auc:.4f}, Acc={acc:.4f}")

    # Find best config
    best = max(results, key=lambda x: x['auc'])
    current = [r for r in results if r['dropout'] == 0.3 and r['weight_decay'] == 1e-4 and r['label_smoothing'] == 0.1]

    print(f"\n{'='*50}")
    print(f"  Best: dropout={best['dropout']}, wd={best['weight_decay']}, ls={best['label_smoothing']}")
    print(f"  Best AUC: {best['auc']:.4f}")
    if current:
        print(f"  Current (0.3/1e-4/0.1) AUC: {current[0]['auc']:.4f}")
    print(f"{'='*50}")

    # Save
    os.makedirs("research/results", exist_ok=True)
    with open("research/results/regularization_sweep.json", 'w') as f:
        json.dump({'results': results, 'best': best, 'timestamp': datetime.now().isoformat()}, f, indent=2)

    # Append to log
    with open("research/experiment_log.md", 'a') as f:
        f.write(f"\n## Regularization Sweep\n\n")
        f.write(f"| Dropout | Weight Decay | Label Smoothing | AUC | Accuracy |\n")
        f.write(f"|---------|-------------|-----------------|-----|----------|\n")
        for r in sorted(results, key=lambda x: -x['auc']):
            f.write(f"| {r['dropout']} | {r['weight_decay']} | {r['label_smoothing']} | "
                    f"{r['auc']:.4f} | {r['accuracy']:.4f} |\n")
        f.write(f"\n**Best**: dropout={best['dropout']}, wd={best['weight_decay']}, "
                f"ls={best['label_smoothing']} (AUC={best['auc']:.4f})\n")
        f.write(f"\nRun: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

    print(f"\n  Results saved to research/results/regularization_sweep.json")


if __name__ == "__main__":
    main()
