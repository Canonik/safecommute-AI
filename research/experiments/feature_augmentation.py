"""
Experiment 5: Feature-Level Augmentation.
- Mixup at feature level (after CNN, before GRU)
- CutMix on spectrograms (swap rectangular regions)
- SpecAugment with adaptive masking
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from safecommute.model import SafeCommuteCNN, ConvBlock, SEBlock
from safecommute.constants import MODEL_SAVE_PATH, DATA_DIR, N_MELS
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from research.experiments.eval_utils import (
    load_stats, get_train_val_loaders, full_evaluation
)
from safecommute.pipeline.train import FocalLoss, compute_class_weights
import random

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6
LABEL_SMOOTHING = 0.1


def cutmix_spectrogram(x, y, alpha=1.0):
    """CutMix on spectrograms: swap rectangular regions between samples."""
    B, C, H, W = x.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B, device=x.device)

    # Random bounding box
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # Adjust lambda by actual area
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    return x_mix, y, y[idx], lam


class SafeCommuteCNN_FeatureMixup(nn.Module):
    """Model with feature-level mixup between CNN and GRU."""

    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)

        freq_after_blocks = n_mels // (2 ** 3)
        self.freq_reduce = nn.Linear(256 * freq_after_blocks, 256)

        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(384, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, feature_mixup=False, mixup_alpha=0.3):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))  # (B, 23, 256)

        # Feature-level mixup: after CNN features, before GRU
        if feature_mixup and self.training:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(B, device=x.device)
            x = lam * x + (1 - lam) * x[idx]

        out, h = self.gru(x)
        h_last = h.squeeze(0)
        h_mean = out.mean(dim=1)
        h_max = out.max(dim=1).values

        x = torch.cat([h_last, h_mean, h_max], dim=1)
        x = self.dropout(x)
        return self.fc(x)


def train_with_cutmix(save_path):
    """Train with CutMix augmentation on spectrograms."""
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SafeCommuteCNN().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)
    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(EPOCHS):
        model.train()
        run_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Strong augmentation
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

            # CutMix with 50% probability
            if np.random.random() < 0.5:
                inputs, labels_a, labels_b, lam = cutmix_spectrogram(inputs, labels)
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

        train_loss = run_loss / len(train_loader)
        train_acc = 100 * correct / total

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

        scheduler.step(epoch)
        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% VL={val_loss_avg:.4f} VA={val_acc:.1f}%", end='')

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

    return model


def train_with_feature_mixup(save_path):
    """Train with feature-level mixup."""
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SafeCommuteCNN_FeatureMixup().to(device)
    class_wts = compute_class_weights(train_dataset).to(device)
    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(EPOCHS):
        model.train()
        run_loss = correct = total = 0

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

            outputs = model(inputs, feature_mixup=True)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = outputs.detach().argmax(1)
            run_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = run_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for v_inp, v_lab in val_loader:
                v_inp, v_lab = v_inp.to(device), v_lab.to(device)
                v_out = model(v_inp, feature_mixup=False)
                val_loss += criterion(v_out, v_lab).item()
                v_correct += (v_out.argmax(1) == v_lab).sum().item()
                v_total += v_lab.size(0)

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_acc = 100 * v_correct / v_total

        scheduler.step(epoch)
        print(f"  E{epoch+1:>2} TL={train_loss:.4f} TA={train_acc:.1f}% VL={val_loss_avg:.4f} VA={val_acc:.1f}%", end='')

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

    return model


def main():
    os.makedirs("research/results", exist_ok=True)

    # Sub-experiment A: CutMix on spectrograms
    print("=== Feature Augmentation: CutMix ===")
    cutmix_path = "research/results/cutmix_model.pth"
    train_with_cutmix(cutmix_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cm = SafeCommuteCNN().to(device)
    model_cm.load_state_dict(torch.load(cutmix_path, map_location=device, weights_only=True))
    full_evaluation(model_cm, device, "CutMix Augmentation", "CutMix 50% prob on spectrograms")

    # Sub-experiment B: Feature-level mixup
    print("\n=== Feature Augmentation: Feature Mixup ===")
    fmix_path = "research/results/feature_mixup_model.pth"
    train_with_feature_mixup(fmix_path)

    model_fm = SafeCommuteCNN_FeatureMixup().to(device)
    model_fm.load_state_dict(torch.load(fmix_path, map_location=device, weights_only=True))
    full_evaluation(model_fm, device, "Feature-Level Mixup", "Mixup after CNN, before GRU")


if __name__ == "__main__":
    main()
