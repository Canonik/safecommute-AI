"""
Shared dataset class for SafeCommute AI.

Provides the PyTorch Dataset that loads pre-computed .pt spectrogram
tensors at training and evaluation time. This is the BASE augmentation
layer — SpecAugment (frequency/time masking) is applied here per-sample.

The two-layer augmentation strategy:
  Layer 1 (here, dataset.py): SpecAugment in __getitem__, applied per-sample
    with 50% probability. Varies every epoch because it runs on-the-fly.
  Layer 2 (train.py, --strong-aug): Batch-level GPU ops (noise, time shift,
    freq dropout). Applied on top of Layer 1 for additional regularization.

Key design decision: .pt files store CLEAN spectrograms. Augmentation is
never baked into saved features. This means:
  - Validation/test sets see clean data (augment=False)
  - Training sees different augmentations each epoch (better generalization)
  - Re-running the pipeline produces identical .pt files (reproducibility)
"""

import os

import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from safecommute.constants import TIME_FRAMES


class TensorAudioDataset(Dataset):
    """
    Loads pre-computed .pt spectrogram tensors from a split directory.

    Expects directory structure: split_dir/{0_safe, 1_unsafe}/*.pt
    where label is inferred from the subdirectory name (0=safe, 1=unsafe).

    Args:
        split_dir: Path to a split directory (e.g., prepared_data/train).
        mean: Global spectrogram mean from training set (for normalization).
        std: Global spectrogram std from training set.
        augment: If True, apply SpecAugment (training only).
        load_teacher: If True, also load teacher soft labels for knowledge
                      distillation (experimental, not used in v2 pipeline).
    """

    def __init__(self, split_dir, mean=0.0, std=1.0, augment=False, load_teacher=False):
        self.filepaths = []
        self.labels = []
        self.mean = mean
        self.std = std
        self.augment = augment
        self.load_teacher = load_teacher

        # SpecAugment: mask up to 10 mel bins (frequency) and 20 time frames.
        # These values are conservative — aggressive masking (e.g., 30 bins)
        # can erase the very frequency bands that distinguish screams from speech.
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=20)

        # Enumerate class directories. sorted() ensures deterministic ordering
        # across platforms for reproducible dataset indices.
        for label, class_name in enumerate(['0_safe', '1_unsafe']):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in sorted(os.listdir(class_dir)):
                if file.endswith('.pt') and not file.endswith('_teacher.pt'):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        features = torch.load(self.filepaths[idx], weights_only=True)

        # Enforce fixed time dimension — some sources may produce slightly
        # different frame counts due to rounding in librosa's STFT
        t = features.shape[-1]
        if t > TIME_FRAMES:
            features = features[:, :, :TIME_FRAMES]
        elif t < TIME_FRAMES:
            pad = torch.zeros(1, features.shape[1], TIME_FRAMES - t)
            features = torch.cat([features, pad], dim=-1)

        # Normalize using global training-set statistics
        features = (features - self.mean) / (self.std + 1e-8)

        # Training-time augmentation (uses torch.rand for DataLoader worker safety):
        if self.augment:
            # Gain augmentation: random ±10 shift in spectrogram domain.
            # Simulates different mic gains / recording distances. With PCEN
            # features this provides additional robustness beyond PCEN's own
            # adaptive gain control.
            gain_shift = (torch.rand(1).item() - 0.5) * 20  # uniform [-10, +10]
            features = features + gain_shift

            # SpecAugment: frequency and time masking
            if torch.rand(1).item() < 0.5:
                features = self.freq_mask(features)
            if torch.rand(1).item() < 0.5:
                features = self.time_mask(features)

        hard_label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.load_teacher:
            teacher_path = self.filepaths[idx].replace('.pt', '_teacher.pt')
            if os.path.exists(teacher_path):
                teacher_soft = torch.load(teacher_path, weights_only=True)
            else:
                teacher_soft = torch.tensor(-1.0)  # sentinel: no teacher label
            return features, hard_label, teacher_soft

        return features, hard_label
