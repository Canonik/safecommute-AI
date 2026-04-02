"""
Domain-adversarial training for SafeCommute AI.

Assigns domain labels based on data source:
  - domain 0 ("acted"): CREMA-D, RAVDESS, TESS, SAVEE
  - domain 1 ("real"):  YouTube, UrbanSound8K, ESC-50, Violence dataset

The encoder learns to classify safe/unsafe while being unable to
distinguish acted from real-world audio → better generalization.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.domain_adversarial import SafeCommuteDACNN
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH, TIME_FRAMES
from safecommute.utils import seed_everything


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4
PATIENCE = 8
DOMAIN_LOSS_WEIGHT = 0.3  # weight for domain adversarial loss

# Prefixes that indicate "acted" audio
ACTED_PREFIXES = ('rav_', 'cremad_', 'tess_', 'savee_')


class DomainAudioDataset(Dataset):
    """Dataset that returns (features, task_label, domain_label)."""
    def __init__(self, split_dir, mean=0.0, std=1.0):
        self.filepaths = []
        self.labels = []
        self.domains = []
        self.mean = mean
        self.std = std

        for label, class_name in enumerate(['0_safe', '1_unsafe']):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in sorted(os.listdir(class_dir)):
                if file.endswith('.pt') and not file.endswith('_teacher.pt'):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(label)
                    # Domain: 0=acted, 1=real
                    is_acted = any(file.startswith(p) for p in ACTED_PREFIXES)
                    self.domains.append(0 if is_acted else 1)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        features = torch.load(self.filepaths[idx], weights_only=True)
        t = features.shape[-1]
        if t > TIME_FRAMES:
            features = features[:, :, :TIME_FRAMES]
        elif t < TIME_FRAMES:
            pad = torch.zeros(1, features.shape[1], TIME_FRAMES - t)
            features = torch.cat([features, pad], dim=-1)
        features = (features - self.mean) / (self.std + 1e-8)
        return (features,
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.domains[idx], dtype=torch.long))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha,
                             reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def train():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Domain-adversarial training on {device}")

    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    train_ds = DomainAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_ds = DomainAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)

    # Count domain distribution
    acted_train = sum(1 for d in train_ds.domains if d == 0)
    real_train = sum(1 for d in train_ds.domains if d == 1)
    print(f"Train: {len(train_ds)} ({acted_train} acted, {real_train} real)")
    print(f"Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model — load pretrained weights from best model
    model = SafeCommuteDACNN().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        base_sd = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
        model.load_from_base(base_sd)

    # Class weights
    counts = [0, 0]
    for lbl in train_ds.labels:
        counts[lbl] += 1
    total = sum(counts)
    class_wts = torch.tensor([total/(2*c) if c > 0 else 1.0 for c in counts], dtype=torch.float32).to(device)

    task_criterion = FocalLoss(alpha=class_wts, gamma=2.0, label_smoothing=0.1)
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0
    save_path = 'safecommute_domain_adversarial.pth'

    for epoch in range(EPOCHS):
        model.train()
        run_task_loss = run_domain_loss = correct = total_samples = 0

        # Gradually increase domain loss weight (curriculum)
        p = epoch / EPOCHS
        lambda_domain = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # sigmoid schedule
        model.grl.lambda_ = lambda_domain * DOMAIN_LOSS_WEIGHT

        for inputs, task_labels, domain_labels in train_loader:
            inputs = inputs.to(device)
            task_labels = task_labels.to(device)
            domain_labels = domain_labels.to(device)

            task_logits, domain_logits = model(inputs, return_domain=True)
            loss_task = task_criterion(task_logits, task_labels)
            loss_domain = domain_criterion(domain_logits, domain_labels)
            loss = loss_task + DOMAIN_LOSS_WEIGHT * loss_domain

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_task_loss += loss_task.item()
            run_domain_loss += loss_domain.item()
            preds = task_logits.detach().argmax(1)
            correct += (preds == task_labels).sum().item()
            total_samples += task_labels.size(0)

        scheduler.step(epoch)
        train_acc = 100 * correct / total_samples
        avg_task = run_task_loss / len(train_loader)
        avg_domain = run_domain_loss / len(train_loader)

        # Validate (task only)
        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for v_inp, v_lab, _ in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                v_out = model(v_inp)
                val_loss += task_criterion(v_out, v_lab).item()
                v_correct += (v_out.argmax(1) == v_lab).sum().item()
                v_total += v_lab.size(0)

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100 * v_correct / v_total

        print(f"  E{epoch+1:>2} TL={avg_task:.4f} DL={avg_domain:.4f} λ={lambda_domain:.2f} "
              f"TA={train_acc:.1f}% VA={val_acc:.1f}%", end='')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_impro = 0
            torch.save(model.state_dict(), save_path)
            print(" *")
        else:
            epochs_no_impro += 1
            print()
            if epochs_no_impro >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # Evaluate on test set
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    test_ds = DomainAudioDataset(os.path.join(DATA_DIR, 'test'), mean, std)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for t_inp, t_lab, _ in test_loader:
            logits = model(t_inp.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(t_lab.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n  TEST: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

    # Per-source accuracy
    src_stats = {}
    test_dir = os.path.join(DATA_DIR, 'test')
    idx = 0
    for cls_label, cls_name in [(0, '0_safe'), (1, '1_unsafe')]:
        cls_dir = os.path.join(test_dir, cls_name)
        for f in sorted(os.listdir(cls_dir)):
            if not f.endswith('.pt') or f.endswith('_teacher.pt'):
                continue
            src = f.split('_')[0]
            if src in ('yt', 'viol'):
                src = '_'.join(f.split('_')[:2])
            if src not in src_stats:
                src_stats[src] = {'c': 0, 't': 0}
            if all_preds[idx] == all_labels[idx]:
                src_stats[src]['c'] += 1
            src_stats[src]['t'] += 1
            idx += 1

    print('\nPer-source:')
    for src, st in sorted(src_stats.items(), key=lambda x: x[1]['c'] / max(x[1]['t'], 1)):
        print(f'  {src:15}: {st["c"] / max(st["t"], 1):6.1%} ({st["c"]}/{st["t"]})')

    return auc


if __name__ == "__main__":
    train()
