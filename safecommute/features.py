"""
Shared feature extraction for SafeCommute AI.
Used by both the data pipeline (training) and inference.
"""

import random
import numpy as np
import librosa
import torch
import torchaudio.transforms as T

from safecommute.constants import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TIME_FRAMES, TARGET_LENGTH


def pad_or_truncate(y):
    """Pad or randomly crop audio to TARGET_LENGTH samples."""
    if len(y) > TARGET_LENGTH:
        start = random.randint(0, len(y) - TARGET_LENGTH)
        return y[start:start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def extract_features(y, augment=True):
    """
    Log-Mel spectrogram with ref=1.0 (fixed reference) to preserve loudness.
    SpecAugment applied only when augment=True (training).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

    if augment:
        if random.random() < 0.5:
            tensor = T.FrequencyMasking(freq_mask_param=10)(tensor)
        if random.random() < 0.5:
            tensor = T.TimeMasking(time_mask_param=20)(tensor)
    return tensor


def preprocess(audio_buffer, mean, std):
    """
    Feature extraction for inference — produces a ready-to-feed tensor.
    Matches extract_features() output format with normalization applied.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Enforce TIME_FRAMES
    t = tensor.shape[-1]
    if t > TIME_FRAMES:
        tensor = tensor[:, :, :, :TIME_FRAMES]
    elif t < TIME_FRAMES:
        pad = torch.zeros(1, 1, N_MELS, TIME_FRAMES - t)
        tensor = torch.cat([tensor, pad], dim=-1)

    return (tensor - mean) / (std + 1e-8)
