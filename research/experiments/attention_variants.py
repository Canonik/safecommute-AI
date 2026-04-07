"""
Experiment 6: Channel Attention Variants.
Compare SE vs CBAM vs ECA vs no attention.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import DataLoader

from safecommute.model import SEBlock
from safecommute.constants import N_MELS, DATA_DIR
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from research.experiments.eval_utils import (
    load_stats, get_train_val_loaders, full_evaluation
)
from safecommute.pipeline.train import FocalLoss, mixup_batch, compute_class_weights
import random

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6
LABEL_SMOOTHING = 0.1
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.3


class ECABlock(nn.Module):
    """Efficient Channel Attention — even lighter than SE."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size based on channels
        k = int(abs(math.log2(channels) + b) / gamma)
        k = k if k % 2 else k + 1  # ensure odd
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y).view(B, C, 1, 1)
        return x * y


class CBAMChannelAttention(nn.Module):
    """Channel attention module of CBAM."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        return self.sigmoid(avg_out + max_out).view(B, C, 1, 1)


class CBAMSpatialAttention(nn.Module):
    """Spatial attention module of CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    """CBAM = Channel + Spatial attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = CBAMChannelAttention(channels, reduction)
        self.spatial_att = CBAMSpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ConvBlockVariant(nn.Module):
    """ConvBlock with configurable attention mechanism."""
    def __init__(self, in_ch, out_ch, attention='se'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if attention == 'se':
            self.att = SEBlock(out_ch)
        elif attention == 'cbam':
            self.att = CBAMBlock(out_ch)
        elif attention == 'eca':
            self.att = ECABlock(out_ch)
        elif attention == 'none':
            self.att = nn.Identity()
        else:
            raise ValueError(f"Unknown attention: {attention}")

    def forward(self, x, pool=(2, 2)):
        x = self.conv(x)
        x = self.att(x)
        return F.avg_pool2d(x, pool)


class SafeCommuteCNN_Attention(nn.Module):
    """SafeCommuteCNN with configurable attention type."""
    def __init__(self, n_mels=N_MELS, n_classes=2, attention='se'):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlockVariant(1, 64, attention)
        self.block2 = ConvBlockVariant(64, 128, attention)
        self.block3 = ConvBlockVariant(128, 256, attention)

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


def train_attention_variant(attention_type, save_path):
    """Train a model with specific attention type."""
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SafeCommuteCNN_Attention(attention=attention_type).to(device)
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

        train_loss = run_loss / len(train_loader)
        train_acc = 100 * correct / total

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


def main():
    os.makedirs("research/results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for att_type in ['cbam', 'eca', 'none']:
        print(f"\n=== Attention Variant: {att_type.upper()} ===")
        save_path = f"research/results/attention_{att_type}_model.pth"
        train_attention_variant(att_type, save_path)

        model = SafeCommuteCNN_Attention(attention=att_type).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        full_evaluation(model, device, f"Attention: {att_type.upper()}",
                       f"{att_type} attention in ConvBlock")


if __name__ == "__main__":
    main()
