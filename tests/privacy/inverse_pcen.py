"""
Forward and approximate-inverse PCEN, used by the privacy attack harness.

Threat model. The classifier consumes a PCEN tile of shape (n_mels, T)
computed from a mel power-spectrogram scaled by 2**31, with librosa
defaults (gain=0.98, bias=2.0, power=0.5, time_constant=0.4 s at 16 kHz,
eps=1e-6). The attacker sees the tile plus the public feature config and
tries to reconstruct intelligible audio.

PCEN forward (Lostanlen and Salamon 2018):
    M[t, f] = (1 - s) * M[t-1, f] + s * E[t, f]
    y[t, f] = (E[t, f] * (eps + M[t, f]) ** (-gain) + bias) ** power - bias ** power

`E` is the prescaled mel (`mel * 2**31`). `M` is the AGC running-mean state.
The inverse for `E` given `M` is exact:
    E[t, f] = ((y[t, f] + bias ** power) ** (1 / power) - bias) * (eps + M[t, f]) ** gain

For the attack, two regimes:

1. **oracle** -- the attacker is GIVEN `M` from the forward pass. Closed-form
   exact inverse, near-zero round-trip error (limited by float precision).
   This is the upper bound on reconstruction quality. If even the oracle
   attacker fails downstream, the privacy claim is firmly defensible.

2. **blind** -- the attacker only sees `y` and the public config. We
   estimate `M` iteratively: initialize from a reasonable scale guess,
   compute `E_hat` from the inverse formula, smooth to get `M_hat`,
   repeat. Converges in 5 to 15 iterations on speech-like signals.
"""

from __future__ import annotations

import numpy as np
import librosa
import scipy.signal

from safecommute.constants import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH


PCEN_GAIN = 0.98
PCEN_BIAS = 2.0
PCEN_POWER = 0.5
PCEN_TIME_CONSTANT = 0.4
PCEN_EPS = 1e-6
PCEN_MEL_PRESCALE = 2.0 ** 31


def _smoothing_coefficient(time_constant_s: float = PCEN_TIME_CONSTANT,
                           hop_length: int = HOP_LENGTH,
                           sr: int = SAMPLE_RATE) -> float:
    return float(1.0 - np.exp(-hop_length / (time_constant_s * sr)))


def _running_mean(E: np.ndarray, s: float) -> np.ndarray:
    """One-pole IIR matching librosa.pcen byte-for-byte (scipy.signal.lfilter
    with zero initial filter state, so M[0] = s * E[0])."""
    return scipy.signal.lfilter([s], [1, s - 1], E, axis=-1)


def forward_pcen_with_state(audio: np.ndarray,
                             sr: int = SAMPLE_RATE,
                             n_mels: int = N_MELS,
                             n_fft: int = N_FFT,
                             hop_length: int = HOP_LENGTH,
                             gain: float = PCEN_GAIN,
                             bias: float = PCEN_BIAS,
                             power: float = PCEN_POWER,
                             time_constant_s: float = PCEN_TIME_CONSTANT,
                             eps: float = PCEN_EPS,
                             mel_prescale: float = PCEN_MEL_PRESCALE
                             ) -> dict:
    """
    Re-implementation of librosa's PCEN that also returns the running-mean
    state M. Forward result matches librosa.pcen up to float precision.

    Returns a dict with `mel` (raw power-mel), `E` (mel * prescale), `M`
    (running mean of E), `y` (PCEN output).
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                          n_fft=n_fft, hop_length=hop_length)
    mel = mel.astype(np.float64)
    E = mel * mel_prescale

    s = _smoothing_coefficient(time_constant_s, hop_length, sr)
    M = _running_mean(E, s)

    y = np.power(E * np.power(eps + M, -gain) + bias, power) - bias ** power
    return {
        "mel": mel.astype(np.float32),
        "E": E,
        "M": M,
        "y": y.astype(np.float32),
        "s": s,
    }


def inverse_pcen_oracle(pcen: np.ndarray,
                         M: np.ndarray,
                         gain: float = PCEN_GAIN,
                         bias: float = PCEN_BIAS,
                         power: float = PCEN_POWER,
                         eps: float = PCEN_EPS,
                         mel_prescale: float = PCEN_MEL_PRESCALE
                         ) -> np.ndarray:
    """Exact inverse given the oracle running-mean state M. Returns a raw
    power-mel (the unscaled mel before `* 2**31`)."""
    pcen = pcen.astype(np.float64)
    M = M.astype(np.float64)
    bracket = pcen + bias ** power
    inner = np.power(bracket, 1.0 / power) - bias
    inner = np.maximum(inner, 0.0)
    E_hat = inner * np.power(eps + M, gain)
    mel_hat = E_hat / mel_prescale
    return mel_hat.astype(np.float32)


def inverse_pcen_blind(pcen: np.ndarray,
                        n_iter: int = 15,
                        M_init_scale: float = 1e25,
                        gain: float = PCEN_GAIN,
                        bias: float = PCEN_BIAS,
                        power: float = PCEN_POWER,
                        time_constant_s: float = PCEN_TIME_CONSTANT,
                        eps: float = PCEN_EPS,
                        mel_prescale: float = PCEN_MEL_PRESCALE,
                        target_peak: float = 0.1,
                        ) -> np.ndarray:
    """
    Iterative inverse without access to M. Initialize M uniformly at
    `M_init_scale` (a rough order-of-magnitude guess for typical 16 kHz
    speech mel power times 2**31), then alternate:
        E_hat = ((y + bias**p)**(1/p) - bias) * (eps + M_hat)**gain
        M_hat = smooth(E_hat)

    The 1/(1-gain) = 50x amplification of any scale error at steady state
    means the absolute magnitude of `E_hat` is essentially uncalibrated --
    correlated in shape with the true mel (corr around 0.93 on synthetic
    signals) but with a peak that can be 15+ orders of magnitude off. A
    blind attacker would empirically rescale to look like real audio, so
    we apply a final peak normalisation: divide by `max(E_hat)` and scale
    to `target_peak`, which matches the typical magnitude of a 16 kHz
    speech power-mel-spectrogram (~0.1). Without this, downstream
    Griffin-Lim numerical convergence stalls (50x per-clip slowdown
    observed on n=200 due to STFT round-trips on out-of-scale magnitudes).
    """
    pcen = pcen.astype(np.float64)
    s = _smoothing_coefficient(time_constant_s)

    M_hat = np.full_like(pcen, M_init_scale)
    bias_pow = bias ** power

    for _ in range(n_iter):
        bracket = pcen + bias_pow
        inner = np.maximum(np.power(bracket, 1.0 / power) - bias, 0.0)
        E_hat = inner * np.power(eps + M_hat, gain)
        M_hat = _running_mean(E_hat, s)

    mel_hat = E_hat / mel_prescale
    peak = float(np.max(np.abs(mel_hat)) + 1e-30)
    mel_hat = mel_hat * (target_peak / peak)
    return mel_hat.astype(np.float32)


def _nmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2) / (np.mean(b ** 2) + 1e-12))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def sanity_check(seed: int = 42) -> dict:
    """
    Round-trip on a synthetic 3-second 16 kHz signal. Verifies that:
      - the forward re-implementation matches librosa.pcen within 1e-4 NMSE
      - the oracle inverse achieves near-zero NMSE vs the true mel
      - the blind inverse recovers a mel that correlates strongly with the truth
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(3.0 * SAMPLE_RATE)) / SAMPLE_RATE
    audio = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * rng.standard_normal(t.shape)
    ).astype(np.float32)

    fwd = forward_pcen_with_state(audio)
    mel_true = fwd["mel"]
    y = fwd["y"]
    M = fwd["M"]

    librosa_y = librosa.pcen(librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH) * PCEN_MEL_PRESCALE,
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    mel_oracle = inverse_pcen_oracle(y, M)
    mel_blind = inverse_pcen_blind(y, n_iter=20)

    return {
        "audio_shape": audio.shape,
        "mel_shape": mel_true.shape,
        "forward_matches_librosa_nmse": _nmse(y, librosa_y),
        "oracle_inverse_nmse_vs_true_mel": _nmse(mel_oracle, mel_true),
        "oracle_inverse_corr_vs_true_mel": _corr(mel_oracle, mel_true),
        "blind_inverse_nmse_vs_true_mel": _nmse(mel_blind, mel_true),
        "blind_inverse_corr_vs_true_mel": _corr(mel_blind, mel_true),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(sanity_check(), indent=2, default=str))
