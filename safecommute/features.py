"""
Shared feature extraction for SafeCommute AI.

This module is the SINGLE source of truth for converting raw audio waveforms
into PCEN spectrogram tensors. It is used by:
  - data_pipeline.py / prepare_*.py: offline feature extraction
  - inference.py / demo.py: real-time feature extraction from microphone
  - finetune.py: processing Layer 3 deployment-specific ambient audio

Uses PCEN (Per-Channel Energy Normalization) instead of log-mel spectrograms.
PCEN performs adaptive gain control per frequency band, making the model
invariant to microphone gain — the same acoustic event produces the same
spectrogram regardless of mic sensitivity. This is critical for deployment
across heterogeneous hardware without per-device calibration.

Reference: Lostanlen & Salamon, "Per-Channel Energy Normalization: Why and How"
(IEEE Signal Processing Letters, 2018).
"""

import numpy as np
import librosa
import torch

from safecommute.constants import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TIME_FRAMES, TARGET_LENGTH


def pad_or_truncate(y):
    """
    Pad or center-crop audio to exactly TARGET_LENGTH (48000) samples.

    Center-crop (rather than random crop) ensures deterministic output:
    the same input file always produces the same .pt tensor.
    """
    if len(y) > TARGET_LENGTH:
        start = (len(y) - TARGET_LENGTH) // 2
        return y[start:start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def chunk_audio(y, sr=SAMPLE_RATE, chunk_sec=3.0, hop_sec=1.5):
    """
    Chunk long audio into overlapping 3-second windows with 50% overlap.

    For AudioSet clips (10s), yields ~5 chunks per clip. The 50% overlap
    ensures every event is fully captured in at least one chunk.
    """
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    chunks = []
    for start in range(0, len(y) - chunk_len + 1, hop_len):
        chunks.append(y[start:start + chunk_len])
    if not chunks and len(y) > sr:
        chunks.append(pad_or_truncate(y))
    return chunks


def extract_features(y):
    """
    Convert a 3-second audio waveform to a PCEN spectrogram tensor.

    PCEN (Per-Channel Energy Normalization) replaces the traditional
    log-mel spectrogram (power_to_db with ref=1.0) with an adaptive
    gain-controlled representation. This makes the output invariant
    to microphone gain — a scream recorded at 0.01 amplitude produces
    the same spectrogram as one recorded at 1.0 amplitude.

    PCEN components:
      1. Temporal integration (smoothing filter per frequency band)
      2. Adaptive gain control (divide by running average)
      3. Dynamic range compression (power-law nonlinearity)

    The mel spectrogram is scaled by 2^31 before PCEN as recommended
    by librosa docs (PCEN expects integer-range magnitudes).

    Returns:
        Tensor of shape (1, N_MELS, TIME_FRAMES) = (1, 64, 188).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # PCEN: adaptive gain control + compression. gain=0.98, bias=2,
    # power=0.5, time_constant=0.4 are librosa defaults tuned for
    # audio event detection.
    pcen_spec = librosa.pcen(
        mel_spec * (2**31),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    return torch.tensor(pcen_spec, dtype=torch.float32).unsqueeze(0)


def preprocess(audio_buffer, mean, std):
    """
    Feature extraction for real-time inference — produces a normalized,
    batch-ready PCEN tensor from a raw audio buffer.

    Args:
        audio_buffer: Raw float32 audio waveform (3 seconds at 16 kHz).
        mean: Global mean of training PCEN spectrograms (from feature_stats.json).
        std: Global std of training PCEN spectrograms.

    Returns:
        Tensor of shape (1, 1, N_MELS, TIME_FRAMES) = (1, 1, 64, 188).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    pcen_spec = librosa.pcen(
        mel_spec * (2**31),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    tensor = torch.tensor(pcen_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Enforce exactly TIME_FRAMES columns
    t = tensor.shape[-1]
    if t > TIME_FRAMES:
        tensor = tensor[:, :, :, :TIME_FRAMES]
    elif t < TIME_FRAMES:
        pad = torch.zeros(1, 1, N_MELS, TIME_FRAMES - t)
        tensor = torch.cat([tensor, pad], dim=-1)

    return (tensor - mean) / (std + 1e-8)
