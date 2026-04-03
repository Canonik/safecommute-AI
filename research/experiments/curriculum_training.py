"""
Experiment 1: Curriculum Learning.
Train on easy samples first, gradually add harder ones.
Easy = high-confidence predictions from current model; hard = uncertain.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH, DATA_DIR, STATS_PATH
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from research.experiments.eval_utils import (
    load_stats, get_train_val_loaders, full_evaluation
)
from v_3.train_experimental import FocalLoss, spec_augment_strong, mixup_batch, compute_class_weights


BATCH_SIZE = 32
EPOCHS_PER_STAGE = 8
LEARNING_RATE = 3e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
MIXUP_PROB = 0.5


def score_difficulty(model, dataset, device, batch_size=64):
    """Score each sample by difficulty (uncertainty of current model).
    Returns array of confidence scores (higher = easier)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    confidences = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1)
            # Confidence = probability of correct class
            for i in range(labels.size(0)):
                conf = probs[i, labels[i]].item()
                confidences.append(conf)

    return np.array(confidences)


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, use_strong_aug=True, save_path=None):
    """Train for a fixed number of epochs."""
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        run_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if use_strong_aug:
                aug_inputs = []
                for i in range(inputs.size(0)):
                    aug_inputs.append(spec_augment_strong(inputs[i].cpu()).to(device))
                inputs = torch.stack(aug_inputs)

            if np.random.random() < MIXUP_PROB:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels)
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
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = run_loss / len(train_loader)

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
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0

        scheduler.step(epoch)
        print(f"    E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% VL={val_loss_avg:.4f} VA={val_acc:.1f}%", end='')

        if val_loss_avg < best_val_loss and save_path:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), save_path)
            print(" *")
        else:
            print()

    return model


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "research/results/curriculum_model.pth"
    os.makedirs("research/results", exist_ok=True)

    print("=== Curriculum Learning ===")

    # Load teacher model to score difficulty
    teacher = SafeCommuteCNN().to(device)
    teacher.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))

    # Load full training data
    mean, std = load_stats()
    full_train = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Score difficulty
    print("  Scoring sample difficulty...")
    confidences = score_difficulty(teacher, full_train, device)
    sorted_indices = np.argsort(-confidences)  # easiest first

    # Create new model
    model = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(full_train).to(device)
    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Stage 1: Easy 33% (highest confidence)
    n = len(full_train)
    stages = [
        ("Easy (top 33%)", sorted_indices[:n//3]),
        ("Medium (top 66%)", sorted_indices[:2*n//3]),
        ("Full (100%)", sorted_indices),
    ]

    for stage_name, indices in stages:
        print(f"\n  Stage: {stage_name} — {len(indices)} samples")
        subset = Subset(full_train, indices.tolist())
        stage_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        model = train_stage(model, stage_loader, val_loader, criterion, optimizer, scheduler,
                           device, EPOCHS_PER_STAGE, save_path=save_path)

    # Load best checkpoint and evaluate
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    results, breakdown = full_evaluation(model, device, "Curriculum Learning (3-stage)",
                                         "easy→medium→full, 8 epochs/stage")
    return results


if __name__ == "__main__":
    main()
