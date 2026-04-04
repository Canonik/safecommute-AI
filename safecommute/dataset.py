"""Shared dataset class for SafeCommute AI."""

import os
import random

import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from safecommute.constants import TIME_FRAMES


class TensorAudioDataset(Dataset):
    """
    Loads pre-computed .pt spectrogram tensors from a split directory.

    When augment=True (training only), applies per-sample SpecAugment
    (frequency and time masking) in __getitem__. This means each epoch
    sees different augmentation — unlike the old pipeline which baked
    one fixed augmentation into the .pt files.
    """

    def __init__(self, split_dir, mean=0.0, std=1.0, augment=False, load_teacher=False):
        self.filepaths = []
        self.labels = []
        self.mean = mean
        self.std = std
        self.augment = augment
        self.load_teacher = load_teacher

        # SpecAugment transforms (only used when augment=True)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=20)

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

        # Enforce fixed time dimension
        t = features.shape[-1]
        if t > TIME_FRAMES:
            features = features[:, :, :TIME_FRAMES]
        elif t < TIME_FRAMES:
            pad = torch.zeros(1, features.shape[1], TIME_FRAMES - t)
            features = torch.cat([features, pad], dim=-1)

        # Normalize
        features = (features - self.mean) / (self.std + 1e-8)

        # Training-time augmentation: different every epoch
        if self.augment:
            if random.random() < 0.5:
                features = self.freq_mask(features)
            if random.random() < 0.5:
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
