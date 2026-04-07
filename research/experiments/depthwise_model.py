"""
Experiment 7: Depthwise Separable Convolutions.
Replace standard Conv2d with depthwise separable in ConvBlock.
Target: reduce params from 1.83M to ~0.5-0.8M while maintaining accuracy.
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

from safecommute.model import SEBlock
from safecommute.constants import N_MELS, DATA_DIR
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from research.experiments.eval_utils import (
    load_stats, get_train_val_loaders, full_evaluation, count_parameters
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


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                                    groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DepthwiseConvBlock(nn.Module):
    """ConvBlock using depthwise separable convolutions."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x, pool=(2, 2)):
        x = self.conv(x)
        x = self.se(x)
        return F.avg_pool2d(x, pool)


class SafeCommuteCNN_Depthwise(nn.Module):
    """Lightweight version with depthwise separable convolutions."""
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = DepthwiseConvBlock(1, 64)
        self.block2 = DepthwiseConvBlock(64, 128)
        self.block3 = DepthwiseConvBlock(128, 256)

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


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("research/results", exist_ok=True)
    save_path = "research/results/depthwise_model.pth"

    print("=== Depthwise Separable Convolutions ===")

    # Quick param comparison
    from safecommute.model import SafeCommuteCNN
    baseline = SafeCommuteCNN()
    dw_model = SafeCommuteCNN_Depthwise()
    print(f"  Baseline params: {count_parameters(baseline):,}")
    print(f"  Depthwise params: {count_parameters(dw_model):,}")
    print(f"  Reduction: {100*(1 - count_parameters(dw_model)/count_parameters(baseline)):.1f}%")

    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SafeCommuteCNN_Depthwise().to(device)
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

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    full_evaluation(model, device, "Depthwise Separable Conv",
                   f"Params: {count_parameters(model):,}")


if __name__ == "__main__":
    main()
