"""
Reconstruction attack baseline (a): mel power-spectrogram -> waveform via
Griffin-Lim phase estimation. Stateless, classical, no pretrained model.

Called by `run_attack_eval.py` after the PCEN-to-mel step (oracle, blind,
or the mel-baseline ablation that skips PCEN entirely).
"""

from __future__ import annotations

import numpy as np
import librosa

from safecommute.constants import SAMPLE_RATE, N_FFT, HOP_LENGTH


GRIFFIN_LIM_ITERS = 60


def mel_to_audio(mel: np.ndarray,
                  sr: int = SAMPLE_RATE,
                  n_fft: int = N_FFT,
                  hop_length: int = HOP_LENGTH,
                  n_iter: int = GRIFFIN_LIM_ITERS,
                  power: float = 2.0,
                  target_length: int | None = None,
                  ) -> np.ndarray:
    """
    Power-mel -> waveform.

    Args:
        mel: shape (n_mels, T), power spectrogram (non-negative).
        target_length: if given, pad/crop the output to this length in samples.
    """
    mel = np.maximum(mel, 0.0).astype(np.float32)
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
        power=power,
    )
    if target_length is not None:
        if audio.shape[0] < target_length:
            audio = np.pad(audio, (0, target_length - audio.shape[0]))
        else:
            audio = audio[:target_length]
    return audio.astype(np.float32)
