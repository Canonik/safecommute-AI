"""
Shared feature extraction for SafeCommute AI.
Used by both the data pipeline (training) and inference.
"""

import numpy as np
import librosa
import torch

from safecommute.constants import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TIME_FRAMES, TARGET_LENGTH


def pad_or_truncate(y):
    """
    Pad or center-crop audio to TARGET_LENGTH samples.

    Uses deterministic center-crop (not random) so the pipeline
    produces identical .pt files across runs.
    """
    if len(y) > TARGET_LENGTH:
        start = (len(y) - TARGET_LENGTH) // 2
        return y[start:start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def chunk_audio(y, sr=SAMPLE_RATE, chunk_sec=3.0, hop_sec=1.5):
    """
    Chunk long audio into overlapping windows.

    For AudioSet clips (10s), this yields ~5 chunks per clip,
    recovering data that pad_or_truncate would discard.
    """
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    chunks = []
    for start in range(0, len(y) - chunk_len + 1, hop_len):
        chunks.append(y[start:start + chunk_len])
    if not chunks and len(y) > sr:  # at least 1 second
        chunks.append(pad_or_truncate(y))
    return chunks


def extract_features(y):
    """
    Log-Mel spectrogram with ref=1.0 (fixed reference) to preserve loudness.

    Returns a clean (un-augmented) spectrogram tensor.
    All augmentation happens at training time in the DataLoader.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)


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
