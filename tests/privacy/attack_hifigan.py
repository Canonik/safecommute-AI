"""
Reconstruction attack baseline (b): power-mel -> waveform via a pretrained
HiFi-GAN vocoder.

We use SpeechBrain's `tts-hifigan-libritts-16kHz` (off-the-shelf, no
fine-tuning). The vocoder expects 80-channel log-mel; our pipeline produces
64-channel power-mel. We bridge them with a fixed analytical adapter:

    adapter = mel_basis_80 @ pinv(mel_basis_64)        # shape (80, 64)

    mel_80_hat = adapter @ mel_64
    log_mel_80 = log(max(mel_80_hat, eps))

This is the minimum-norm least-squares solution to "find an 80-mel that,
projected through the 80-mel filter bank, best matches the energy
distribution captured by the 64-mel filter bank." No empirical
calibration is needed -- the adapter is closed-form and reviewer-defendable
as part of the public feature config the attacker is assumed to know.

Threat-model framing: the attacker has the public PCEN/mel hyperparameters
plus the published vocoder weights. They build the analytical adapter and
feed the result to the vocoder. We measure how intelligible the output is.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import librosa

from safecommute.constants import SAMPLE_RATE, N_FFT, HOP_LENGTH


HIFIGAN_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
SPEECHBRAIN_CACHE = Path(
    os.environ.get(
        "SAFECOMMUTE_SPEECHBRAIN_CACHE",
        str(Path.home() / ".cache" / "safecommute-ai" / "speechbrain"),
    )
)
HIFIGAN_SAVEDIR = str(SPEECHBRAIN_CACHE / "hifigan-libritts-16kHz")
TARGET_N_MELS = 80
# Match the vocoder's hparams: upsample_factors=[8,8,2,2] => 256x per frame;
# inference_padding=5 mel frames each side that we must trim back off so the
# output time-aligns with the original audio.
HIFIGAN_UPSAMPLE = 8 * 8 * 2 * 2
HIFIGAN_INFERENCE_PADDING_FRAMES = 5


@lru_cache(maxsize=1)
def _build_mel_adapter() -> tuple[np.ndarray, int]:
    """Returns (adapter, n_fft). Adapter shape: (80, 64)."""
    mel_64 = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=64)
    mel_80 = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=TARGET_N_MELS)
    # Closed-form least-squares: project 64-mel back to STFT-power via pinv,
    # then re-project to 80-mel. Pure linear algebra, no training.
    adapter = mel_80 @ np.linalg.pinv(mel_64)
    return adapter.astype(np.float32), N_FFT


def adapt_64_to_80_mel(mel_64_power: np.ndarray) -> np.ndarray:
    """Map a 64-channel power-mel to an 80-channel power-mel."""
    adapter, _ = _build_mel_adapter()
    mel_80 = adapter @ mel_64_power.astype(np.float32)
    return np.maximum(mel_80, 0.0)


@lru_cache(maxsize=1)
def _load_hifigan():
    from speechbrain.inference.vocoders import HIFIGAN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hifigan = HIFIGAN.from_hparams(
        source=HIFIGAN_SOURCE,
        savedir=HIFIGAN_SAVEDIR,
        run_opts={"device": device},
    )
    return hifigan, device


def mel_to_audio(mel_64_power: np.ndarray,
                  target_length: int | None = None) -> np.ndarray:
    """
    Power-64-mel (n_mels=64, T) -> waveform via 80-mel adapter -> HiFi-GAN.

    HiFi-GAN expects natural-log mel features at 80 channels.
    """
    mel_80 = adapt_64_to_80_mel(mel_64_power)
    log_mel = np.log(np.maximum(mel_80, 1e-5)).astype(np.float32)

    hifigan, device = _load_hifigan()
    inp = torch.from_numpy(log_mel).unsqueeze(0).to(device)
    with torch.no_grad():
        wav = hifigan.decode_batch(inp).squeeze().cpu().numpy()

    wav = wav.astype(np.float32)
    # Trim the inference_padding the vocoder adds to each end so the audio
    # time-aligns with the original 3-second mel input.
    pad = HIFIGAN_INFERENCE_PADDING_FRAMES * HIFIGAN_UPSAMPLE
    if wav.shape[0] > 2 * pad:
        wav = wav[pad:-pad]
    if target_length is not None:
        if wav.shape[0] < target_length:
            wav = np.pad(wav, (0, target_length - wav.shape[0]))
        else:
            wav = wav[:target_length]
    return wav
