"""
SOTA Benchmark: Train established audio models on our dataset.
Fair comparison: same data, same splits, same metrics.
Models: PANNs CNN14, Audio Spectrogram Transformer (AST-mini), simple baselines.
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from safecommute.constants import DATA_DIR, N_MELS, TIME_FRAMES
from safecommute.dataset import TensorAudioDataset
from safecommute.utils import seed_everything
from research.experiments.eval_utils import (
    load_stats, get_test_loader, get_train_val_loaders,
    per_source_breakdown, measure_latency,
    count_parameters, model_size_mb, log_experiment, full_evaluation
)
from safecommute.pipeline.train import FocalLoss, compute_class_weights

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 6


# ============================================
# Model 1: Simple CNN Baseline (minimal model)
# ============================================
class SimpleCNN(nn.Module):
    """Bare minimum CNN baseline — no attention, no GRU."""
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)


# ============================================
# Model 2: ResNet-style for Audio
# ============================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AudioResNet(nn.Module):
    """ResNet-style audio classifier with residual connections."""
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(self.dropout(x))


# ============================================
# Model 3: Transformer-based (mini AST)
# ============================================
class PatchEmbedding(nn.Module):
    """Split spectrogram into patches and embed."""
    def __init__(self, n_mels=64, patch_h=8, patch_w=8, embed_dim=128):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(patch_h, patch_w),
                             stride=(patch_h, patch_w))

    def forward(self, x):
        # x: (B, 1, 64, 188)
        x = self.proj(x)  # (B, embed_dim, H/ph, W/pw)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MiniAST(nn.Module):
    """Mini Audio Spectrogram Transformer — 2 layers, 4 heads."""
    def __init__(self, n_mels=64, n_classes=2, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_mels, patch_h=8, patch_w=8, embed_dim=embed_dim)

        # Calculate number of patches
        n_patches = (n_mels // 8) * (TIME_FRAMES // 8)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # Pad time dimension to be divisible by 8
        B, C, H, W = x.shape
        if W % 8 != 0:
            pad_w = 8 - (W % 8)
            x = F.pad(x, (0, pad_w))

        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Adjust pos_embed if needed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Handle position embedding size mismatch
        if x.size(1) != self.pos_embed.size(1):
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=x.size(1), mode='linear', align_corners=False
            ).permute(0, 2, 1)
            x = x + pos
        else:
            x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)


# ============================================
# Model 4: LSTM baseline (sequential model)
# ============================================
class AudioLSTM(nn.Module):
    """LSTM-only baseline on mel frames."""
    def __init__(self, n_mels=N_MELS, n_classes=2, hidden=128):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.lstm = nn.LSTM(n_mels, hidden, num_layers=2, batch_first=True,
                           dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 1, 64, 188)
        x = self.bn(x)
        x = x.squeeze(1).permute(0, 2, 1)  # (B, 188, 64)
        out, _ = self.lstm(x)  # (B, 188, 256)
        x = out.mean(dim=1)  # Global average pool
        return self.fc(self.dropout(x))


def train_model(model, model_name, device, save_path):
    """Train a model with the same recipe as our production model."""
    mean, std = load_stats()
    train_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'train'), mean, std)
    val_dataset = TensorAudioDataset(os.path.join(DATA_DIR, 'val'), mean, std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_wts = compute_class_weights(train_dataset).to(device)
    criterion = FocalLoss(alpha=class_wts, gamma=3.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float('inf')
    epochs_no_impro = 0

    for epoch in range(EPOCHS):
        model.train()
        run_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    return model


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("research/results", exist_ok=True)

    models = [
        ("Simple CNN (baseline)", SimpleCNN(), "research/results/simple_cnn.pth"),
        ("Audio ResNet", AudioResNet(), "research/results/audio_resnet.pth"),
        ("Mini AST (2L/4H)", MiniAST(), "research/results/mini_ast.pth"),
        ("BiLSTM", AudioLSTM(), "research/results/audio_lstm.pth"),
    ]

    for name, model, save_path in models:
        print(f"\n{'='*50}")
        print(f"  SOTA Benchmark: {name}")
        print(f"  Params: {count_parameters(model):,}")
        print(f"{'='*50}")

        model = model.to(device)
        model = train_model(model, name, device, save_path)
        full_evaluation(model, device, f"SOTA: {name}",
                       f"Params={count_parameters(model):,}")


if __name__ == "__main__":
    main()
