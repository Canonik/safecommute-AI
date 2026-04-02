"""
Domain-adversarial training for SafeCommute AI.

Adds a gradient reversal layer + domain classifier to make the encoder
learn features that are domain-invariant (can't distinguish acted vs real audio).

Based on: "Domain Adversarial for Acoustic Emotion Recognition"
(Abdelwahab & Busso, IEEE TASLP 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from safecommute.model import SafeCommuteCNN, ConvBlock, SEBlock
from safecommute.constants import N_MELS


class GradientReversalFunction(Function):
    """Reverses gradient during backward pass (for adversarial training)."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class SafeCommuteDACNN(nn.Module):
    """
    Domain-Adversarial SafeCommuteCNN.

    Same encoder as SafeCommuteCNN but with an additional domain classifier
    head that tries to predict whether audio is "acted" or "real-world".
    The gradient reversal layer ensures the encoder learns to CONFUSE the
    domain classifier → domain-invariant features.
    """
    def __init__(self, n_mels=N_MELS, n_classes=2, n_domains=2):
        super().__init__()
        # Shared encoder (identical to SafeCommuteCNN)
        self.bn_input = nn.BatchNorm2d(1)
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)

        freq_after_blocks = n_mels // (2 ** 3)
        self.freq_reduce = nn.Linear(256 * freq_after_blocks, 256)
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)

        # Task classifier (safe/unsafe)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(384, n_classes)

        # Domain classifier with gradient reversal
        self.grl = GradientReversal(lambda_=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_domains),
        )

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

    def _encode(self, x):
        """Shared encoder → 384-dim feature vector."""
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
        return torch.cat([h_last, h_mean, h_max], dim=1)

    def forward(self, x, return_domain=False):
        features = self._encode(x)
        task_logits = self.fc(self.dropout(features))

        if return_domain:
            reversed_features = self.grl(features)
            domain_logits = self.domain_classifier(reversed_features)
            return task_logits, domain_logits
        return task_logits

    def load_from_base(self, base_state_dict):
        """Load weights from a trained SafeCommuteCNN checkpoint."""
        own_state = self.state_dict()
        loaded = 0
        for name, param in base_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"  Loaded {loaded}/{len(base_state_dict)} weights from base model")
