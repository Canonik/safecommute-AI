"""
Shared feature extraction for SafeCommute AI.

This module is the SINGLE source of truth for converting raw audio waveforms
into log-mel spectrogram tensors. It is used by:
  - data_pipeline.py / prepare_*.py: offline feature extraction (clean, no augmentation)
  - inference.py: real-time feature extraction from microphone buffer
  - finetune.py: processing Layer 3 deployment-specific ambient audio

All functions here produce CLEAN (un-augmented) spectrograms. Augmentation
(SpecAugment, noise injection, mixup) is applied at training time only,
in dataset.py and train.py. This separation ensures that saved .pt files
are deterministic and augmentation varies across epochs.
"""

import numpy as np
import librosa
import torch

from safecommute.constants import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TIME_FRAMES, TARGET_LENGTH


def pad_or_truncate(y):
    """
    Pad or center-crop audio to exactly TARGET_LENGTH (48000) samples.

    Center-crop (rather than random crop) ensures deterministic output:
    the same input file always produces the same .pt tensor, which is
    critical for reproducible splits and leakage verification.

    Zero-padding short clips is preferred over stretching because time-
    stretching distorts pitch and temporal patterns that the model relies
    on for threat detection.
    """
    if len(y) > TARGET_LENGTH:
        # Center-crop preserves the middle of the clip, which typically
        # contains the strongest signal (AudioSet labels mark 10s segments
        # where the event occurs throughout, but edges may have silence).
        start = (len(y) - TARGET_LENGTH) // 2
        return y[start:start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def chunk_audio(y, sr=SAMPLE_RATE, chunk_sec=3.0, hop_sec=1.5):
    """
    Chunk long audio into overlapping 3-second windows with 50% overlap.

    Args:
        y: Audio waveform (numpy array).
        sr: Sample rate (default: 16 kHz).
        chunk_sec: Window length in seconds (matches model input duration).
        hop_sec: Hop between windows. 1.5s (50% overlap) provides 2x data
                 multiplier and ensures no event is split across a boundary
                 without being fully captured in at least one chunk.

    For AudioSet clips (10s), this yields ~5 chunks per clip, recovering
    data that pad_or_truncate would discard. For YouTube/violence clips
    (variable length), it adapts automatically.

    Returns:
        List of numpy arrays, each exactly chunk_len samples.
    """
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    chunks = []
    for start in range(0, len(y) - chunk_len + 1, hop_len):
        chunks.append(y[start:start + chunk_len])
    if not chunks and len(y) > sr:  # at least 1 second of audio
        chunks.append(pad_or_truncate(y))
    return chunks


def extract_features(y):
    """
    Convert a 3-second audio waveform to a log-mel spectrogram tensor.

    Uses ref=1.0 (absolute reference) rather than librosa's default ref=np.max,
    which normalizes each spectrogram to its own peak. With ref=1.0, loudness
    information is preserved across samples — crucial because threat sounds
    (screams, gunshots) are typically louder than background noise, and this
    amplitude difference is a useful discriminative feature.

    Returns:
        Tensor of shape (1, N_MELS, TIME_FRAMES) = (1, 64, 188).
        The leading dimension is the channel dim (mono = 1 channel).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)


def preprocess(audio_buffer, mean, std):
    """
    Feature extraction for real-time inference — produces a normalized,
    batch-ready tensor from a raw audio buffer.

    Mirrors extract_features() but adds:
      1. Batch dimension (unsqueeze to 4D for model input)
      2. Time-axis enforcement (pad/truncate to exactly TIME_FRAMES)
      3. Global normalization using training-set statistics

    Args:
        audio_buffer: Raw float32 audio waveform (3 seconds at 16 kHz).
        mean: Global mean of training spectrograms (from feature_stats.json).
        std: Global std of training spectrograms.

    Returns:
        Tensor of shape (1, 1, N_MELS, TIME_FRAMES) = (1, 1, 64, 188).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Enforce exactly TIME_FRAMES columns — edge cases from slight buffer
    # length mismatches in real-time audio capture
    t = tensor.shape[-1]
    if t > TIME_FRAMES:
        tensor = tensor[:, :, :, :TIME_FRAMES]
    elif t < TIME_FRAMES:
        pad = torch.zeros(1, 1, N_MELS, TIME_FRAMES - t)
        tensor = torch.cat([tensor, pad], dim=-1)

    # Normalize to zero-mean unit-variance using training-set statistics.
    # Epsilon prevents division by zero if std is very small.
    return (tensor - mean) / (std + 1e-8)
