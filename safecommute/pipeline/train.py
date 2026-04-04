"""
Training script for SafeCommute AI.
Supports: focal loss, cosine annealing, batch GPU augmentation, mixup.

Augmentation strategy:
  - Base augmentation: SpecAugment (freq/time masking) in dataset.__getitem__
    Applied per-sample, different every epoch. Always on for training.
  - Strong augmentation (--strong-aug): batch-level GPU ops in training loop
    Gaussian noise, circular time shift, frequency band dropout.
    Applied on top of base augmentation for extra regularization.
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.dataset import TensorAudioDataset
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH, THRESHOLDS_PATH
from safecommute.utils import seed_everything, worker_init_fn

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
MIXUP_PROB = 0.5


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance and hard examples."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                  reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[idx]
    return x_m, y, y[idx], lam


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


def compute_class_weights(dataset):
    counts = [0, 0]
    for lbl in dataset.labels:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train(use_focal=False, use_cosine=False, use_strong_aug=False, gamma=2.0,
          save_path=None, seed=42):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = save_path or MODEL_SAVE_PATH

    mean, std = load_stats()

    # Training dataset gets base augmentation (SpecAugment in __getitem__)
    train_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'train'), mean, std, augment=True)
    val_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'val'), mean, std, augment=False)

    if len(train_dataset) == 0:
        print("Error: No training data.")
        return None

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Focal={use_focal}(gamma={gamma}), Cosine={use_cosine}, "
          f"StrongAug={use_strong_aug}, Seed={seed}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)

    if use_focal:
        criterion = FocalLoss(alpha=class_wts, gamma=gamma,
                              label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_wts,
                                        label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(EPOCHS):
        model.train()
        run_loss = correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Strong augmentation: per-sample GPU ops (no CPU transfer)
            if use_strong_aug:
                B = inputs.size(0)
                # Gaussian noise: per-sample mask (30% of samples)
                noise_mask = torch.rand(B, 1, 1, 1, device=inputs.device) < 0.3
                inputs = inputs + noise_mask * torch.randn_like(inputs) * 0.1
                # Circular time shift: per-sample random shift (30% of samples)
                for i in range(B):
                    if random.random() < 0.3:
                        shift = random.randint(-20, 20)
                        inputs[i] = torch.roll(inputs[i], shifts=shift, dims=-1)
                # Frequency band dropout: per-sample (20% of samples)
                for i in range(B):
                    if random.random() < 0.2:
                        f_start = random.randint(0, 50)
                        f_width = random.randint(3, 10)
                        inputs[i, :, f_start:f_start + f_width, :] = 0

            if np.random.random() < MIXUP_PROB:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                preds = outputs.detach().argmax(1)
                c_lab = labels_a if lam >= 0.5 else labels_b
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.detach().argmax(1)
                c_lab = labels

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            correct += (preds == c_lab).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = run_loss / len(train_loader)

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
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0

        if use_cosine:
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss_avg)

        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% "
              f"VL={val_loss_avg:.4f} VA={val_acc:.1f}%", end='')

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

    # Evaluate AUC on test set
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    test_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'test'), mean, std, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for t_inp, t_lab in test_loader:
            logits = model(t_inp.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(t_lab.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n  TEST: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--focal', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--strong-aug', action='store_true')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(use_focal=args.focal, use_cosine=args.cosine, use_strong_aug=args.strong_aug,
          gamma=args.gamma, save_path=args.save, seed=args.seed)
