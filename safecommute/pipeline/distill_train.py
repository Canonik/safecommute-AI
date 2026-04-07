"""
Knowledge distillation training for SafeCommute AI.

Trains SafeCommuteCNN using soft labels from PANNs CNN14 (80M params,
trained on 2M AudioSet clips). The student learns from both:
  - Hard labels: binary safe/unsafe ground truth
  - Soft labels: PANNs probability distribution over safe/unsafe

The soft labels encode PANNs' confidence and inter-class similarity,
providing richer gradient signal than binary labels alone. For example,
PANNs might label a laughter clip as "90% safe, 10% scream-like" —
this teaches the student that laughter shares features with screams.

Prerequisites:
    PYTHONPATH=. python -m safecommute.distill   # generate _teacher.pt files

Usage:
    PYTHONPATH=. python safecommute/pipeline/distill_train.py \\
        --kd-temperature 4.0 --kd-alpha 0.7 --cosine --strong-aug --seed 42
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
from safecommute.constants import DATA_DIR, STATS_PATH, MODEL_SAVE_PATH
from safecommute.utils import seed_everything, worker_init_fn
from safecommute.pipeline.train import FocalLoss, mixup_batch, compute_class_weights

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6


def distillation_loss(student_logits, teacher_logits, hard_labels, T, alpha):
    """
    Combined knowledge distillation + hard label loss.

    KD loss: KL divergence between softened student and teacher distributions.
    Temperature T softens both distributions, making the teacher's uncertainty
    visible to the student. Higher T → softer distributions → more knowledge
    transfer from the teacher's "dark knowledge" (inter-class similarities).

    Total loss = α * KD_loss * T² + (1 - α) * CE_loss

    The T² scaling compensates for the gradient magnitude reduction caused
    by the temperature (Hinton et al., 2015).
    """
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    ce_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss


def load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    with open(STATS_PATH) as f:
        s = json.load(f)
    return s['mean'], s['std']


def train_distill(kd_temperature=4.0, kd_alpha=0.7, use_cosine=False,
                  use_strong_aug=False, save_path=None, seed=42):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = save_path or MODEL_SAVE_PATH

    mean, std = load_stats()

    # Load with teacher soft labels
    train_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'train'), mean, std, augment=True, load_teacher=True)
    val_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'val'), mean, std, augment=False, load_teacher=True)

    # Count how many samples have teacher labels
    n_with_teacher = sum(1 for i in range(len(train_dataset))
                         if os.path.exists(train_dataset.filepaths[i].replace('.pt', '_teacher.pt')))
    n_total = len(train_dataset)
    print(f"Train: {n_total} total, {n_with_teacher} with teacher labels ({100*n_with_teacher/n_total:.0f}%)")

    if n_with_teacher == 0:
        print("ERROR: No teacher labels found. Run: PYTHONPATH=. python -m safecommute.distill")
        return None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    model = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)

    print(f"KD: T={kd_temperature}, alpha={kd_alpha}")
    print(f"Cosine={use_cosine}, StrongAug={use_strong_aug}, Seed={seed}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)

    # CE loss for samples without teacher labels
    ce_criterion = nn.CrossEntropyLoss(weight=class_wts)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(EPOCHS):
        model.train()
        run_loss = correct = total = kd_count = ce_count = 0

        for batch in train_loader:
            features, hard_labels, teacher_soft = batch
            features = features.to(device)
            hard_labels = hard_labels.to(device)
            teacher_soft = teacher_soft.to(device)

            # Strong augmentation (batch-level GPU ops)
            if use_strong_aug:
                B = features.size(0)
                noise_mask = torch.rand(B, 1, 1, 1, device=device) < 0.3
                features = features + noise_mask * torch.randn_like(features) * 0.1
                for i in range(B):
                    if random.random() < 0.3:
                        shift = random.randint(-20, 20)
                        features[i] = torch.roll(features[i], shifts=shift, dims=-1)
                    if random.random() < 0.2:
                        f_start = random.randint(0, 50)
                        f_width = random.randint(3, 10)
                        features[i, :, f_start:f_start + f_width, :] = 0

            outputs = model(features)

            # Split batch: samples WITH teacher labels vs WITHOUT
            has_teacher = teacher_soft[:, 0] != -1.0  # sentinel check
            if has_teacher.any():
                kd_out = outputs[has_teacher]
                kd_teacher = teacher_soft[has_teacher]
                kd_hard = hard_labels[has_teacher]
                loss_kd = distillation_loss(
                    kd_out, kd_teacher, kd_hard, kd_temperature, kd_alpha)
                kd_count += has_teacher.sum().item()
            else:
                loss_kd = torch.tensor(0.0, device=device)

            no_teacher = ~has_teacher
            if no_teacher.any():
                ce_out = outputs[no_teacher]
                ce_hard = hard_labels[no_teacher]
                loss_ce = ce_criterion(ce_out, ce_hard)
                ce_count += no_teacher.sum().item()
            else:
                loss_ce = torch.tensor(0.0, device=device)

            # Combined loss (weighted by count)
            n_kd = has_teacher.sum().float()
            n_ce = no_teacher.sum().float()
            n_total_batch = n_kd + n_ce
            if n_total_batch > 0:
                loss = (n_kd * loss_kd + n_ce * loss_ce) / n_total_batch
            else:
                loss = loss_kd

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item()
            preds = outputs.detach().argmax(1)
            correct += (preds == hard_labels).sum().item()
            total += hard_labels.size(0)

        train_acc = 100 * correct / total
        train_loss = run_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                features, hard_labels, teacher_soft = batch
                features = features.to(device)
                hard_labels = hard_labels.to(device)
                teacher_soft = teacher_soft.to(device)
                v_out = model(features)

                has_teacher = teacher_soft[:, 0] != -1.0
                if has_teacher.any():
                    vl = distillation_loss(
                        v_out[has_teacher], teacher_soft[has_teacher],
                        hard_labels[has_teacher], kd_temperature, kd_alpha)
                else:
                    vl = ce_criterion(v_out, hard_labels)
                val_loss += vl.item()
                v_correct += (v_out.argmax(1) == hard_labels).sum().item()
                v_total += hard_labels.size(0)

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0

        if use_cosine:
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss_avg)

        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% "
              f"VL={val_loss_avg:.4f} VA={val_acc:.1f}% "
              f"(KD:{kd_count} CE:{ce_count})", end='')

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

    # Test evaluation
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    test_dataset = TensorAudioDataset(
        os.path.join(DATA_DIR, 'test'), mean, std, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    auc = roc_auc_score(all_labels, all_probs)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='weighted')
    print(f"\n  TEST: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

    # Probability calibration check
    safe_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
    unsafe_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]
    print(f"  Safe mean={np.mean(safe_probs):.3f} Unsafe mean={np.mean(unsafe_probs):.3f} "
          f"Separation={np.mean(unsafe_probs)-np.mean(safe_probs):.3f}")

    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Knowledge distillation training')
    parser.add_argument('--kd-temperature', type=float, default=4.0)
    parser.add_argument('--kd-alpha', type=float, default=0.7)
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--strong-aug', action='store_true')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_distill(kd_temperature=args.kd_temperature, kd_alpha=args.kd_alpha,
                  use_cosine=args.cosine, use_strong_aug=args.strong_aug,
                  save_path=args.save, seed=args.seed)
