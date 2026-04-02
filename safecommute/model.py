"""
Canonical model definition for SafeCommute AI.

Architecture: CNN6-style encoder + SE attention + GRU with multi-scale pooling.

Based on PANNs (Kong et al. 2020) — double-conv blocks designed for mel
spectrograms rather than ImageNet-pretrained networks. The GRU captures
temporal escalation patterns (rising pitch, increasing intensity) that
a single-frame CNN cannot detect.

Squeeze-and-Excitation (SE) blocks add channel attention at negligible
parameter cost, letting the model learn which frequency bands matter most.

Multi-scale temporal pooling concatenates the GRU's last hidden state
with mean and max pooling over the full output sequence, capturing both
endpoint and aggregate statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from safecommute.constants import N_MELS


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excite(w).view(B, C, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Two conv layers with BN+ReLU, followed by average pooling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x, pool=(2, 2)):
        x = self.conv(x)
        x = self.se(x)
        return F.avg_pool2d(x, pool)


class SafeCommuteCNN(nn.Module):
    """
    CNN6-style encoder + SE attention + GRU with multi-scale pooling.

    Tensor flow (B = batch, Fr = mel bins, T = time frames):
        Input          (B,   1,  64, 188)
        block1 pool2x2 (B,  64,  32,  94)   + SE attention
        block2 pool2x2 (B, 128,  16,  47)   + SE attention
        block3 pool2x2 (B, 256,   8,  23)   + SE attention
        reshape+proj   (B,  23, 256)         Linear(2048→256)
        GRU            (B,  23, 128)         output sequence
        multi-pool     (B, 384)              last + mean + max
        FC             (B,   2)
    """
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)

        freq_after_blocks = n_mels // (2 ** 3)  # 64 // 8 = 8
        self.freq_reduce = nn.Linear(256 * freq_after_blocks, 256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        # Multi-scale pooling: last_hidden + mean + max = 384
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
        # x: (B, 1, 64, 188)
        x = self.bn_input(x)
        x = self.block1(x)   # (B, 64, 32, 94)
        x = self.block2(x)   # (B, 128, 16, 47)
        x = self.block3(x)   # (B, 256, 8, 23)

        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)  # (B, 23, 2048)
        x = F.relu(self.freq_reduce(x))                    # (B, 23, 256)

        out, h = self.gru(x)          # out: (B, 23, 128), h: (1, B, 128)
        h_last = h.squeeze(0)         # (B, 128)
        h_mean = out.mean(dim=1)      # (B, 128)
        h_max = out.max(dim=1).values # (B, 128)

        x = torch.cat([h_last, h_mean, h_max], dim=1)  # (B, 384)
        x = self.dropout(x)
        return self.fc(x)             # (B, 2)
