"""SafeCommute AI — standalone CPU inference demo (PyTorch-free).

Takes a .wav path, runs the INT8 ONNX model with PCEN preprocessing,
prints safe/unsafe + probability + inference latency.

Dependencies (all pure-python or C-wheel, NO PyTorch):
    pip install numpy scipy librosa onnxruntime

Usage:
    python infer.py sample.wav
    python infer.py --model safecommute_v2_int8.onnx sample.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import librosa
import numpy as np
import onnxruntime as ort
import scipy.io.wavfile

SAMPLE_RATE = 16000
DURATION_SEC = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)  # 48000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TIME_FRAMES = 188

# Feature normalization — must match feature_stats.json (shipped alongside).
DEFAULT_MEAN = 0.4967380464076996
DEFAULT_STD = 0.6738999485969543


def load_wav(path: str) -> np.ndarray:
    """Load a WAV at 16 kHz mono as float32 in [-1, 1]."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def pad_or_center_crop(y: np.ndarray) -> np.ndarray:
    """Pad with zeros or center-crop to exactly 3 s at 16 kHz."""
    if len(y) > TARGET_LENGTH:
        start = (len(y) - TARGET_LENGTH) // 2
        return y[start:start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def extract_pcen(y: np.ndarray) -> np.ndarray:
    """Raw audio → PCEN spectrogram (1, 1, 64, 188) float32.

    Matches safecommute/features.py exactly: 64 mel bins, n_fft=1024,
    hop=256, 2^31 scaling, librosa PCEN defaults (gain=0.98, bias=2,
    power=0.5, time_constant=0.4).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    pcen = librosa.pcen(
        mel * (2 ** 31),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    spec = pcen.astype(np.float32)
    if spec.shape[-1] > TIME_FRAMES:
        spec = spec[:, :TIME_FRAMES]
    elif spec.shape[-1] < TIME_FRAMES:
        pad = np.zeros((N_MELS, TIME_FRAMES - spec.shape[-1]), dtype=np.float32)
        spec = np.concatenate([spec, pad], axis=-1)
    return spec[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 188)


def normalize(spec: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((spec - mean) / (std + 1e-8)).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('wav', help='Path to a .wav file (any sample rate, mono or stereo)')
    p.add_argument('--model',
                   default=os.path.join(here, 'safecommute_v2_int8.onnx'),
                   help='ONNX model path (default: INT8 next to this script)')
    p.add_argument('--stats',
                   default=os.path.join(here, 'feature_stats.json'),
                   help='feature_stats.json (mean/std)')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Decision threshold on P(unsafe) (default 0.5). '
                        'For per-site fine-tuned models use the '
                        'low_fpr value from thresholds.json.')
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"error: model not found at {args.model}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.wav):
        print(f"error: wav not found at {args.wav}", file=sys.stderr)
        sys.exit(2)

    # Load normalization stats (falls back to repo defaults if file missing).
    if os.path.exists(args.stats):
        with open(args.stats) as f:
            stats = json.load(f)
        mean = float(stats.get('mean', DEFAULT_MEAN))
        std = float(stats.get('std', DEFAULT_STD))
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD

    t0 = time.perf_counter()
    y = load_wav(args.wav)
    y = pad_or_center_crop(y)
    t_load = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    spec = extract_pcen(y)
    spec = normalize(spec, mean, std)
    t_pre = (time.perf_counter() - t1) * 1000

    sess = ort.InferenceSession(args.model,
                                providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    t2 = time.perf_counter()
    logits = sess.run(None, {in_name: spec})[0]
    t_model = (time.perf_counter() - t2) * 1000

    probs = softmax(logits)[0]
    p_unsafe = float(probs[1])
    label = 'UNSAFE' if p_unsafe >= args.threshold else 'safe'

    print(f"file           : {args.wav}")
    print(f"model          : {os.path.basename(args.model)}")
    print(f"P(unsafe)      : {p_unsafe:.3f}   "
          f"(threshold {args.threshold:.2f})")
    print(f"prediction     : {label}")
    print(f"latency (ms)   : load {t_load:.1f}  preprocess {t_pre:.1f}  "
          f"model {t_model:.1f}  total {t_load+t_pre+t_model:.1f}")


if __name__ == '__main__':
    main()
