#!/usr/bin/env python3
"""
SafeCommute AI — Live Demo

Records from your microphone and classifies each 3-second window as
Safe/Warning/Alert in real-time. No audio is saved to disk.

Uses PCEN (Per-Channel Energy Normalization) for gain-invariant features.
No mic calibration needed — PCEN adapts to any microphone gain level.

Usage:
    PYTHONPATH=. python demo.py
    PYTHONPATH=. python demo.py --model models/metro_model.pth
"""

import sys
import os
import json
import time
import argparse
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import pyaudio
from safecommute.model import SafeCommuteCNN
from safecommute.features import preprocess
from safecommute.constants import SAMPLE_RATE, MODEL_SAVE_PATH, STATS_PATH

WINDOW_SEC = 3
STRIDE_SEC = 1
CHUNK = int(SAMPLE_RATE * STRIDE_SEC)
BUFFER = int(SAMPLE_RATE * WINDOW_SEC)
SMOOTHING = 4
ENERGY_GATE_RMS = 0.003
SPEECH_THRESH_BOOST = 0.85


def detect_speech(audio, sr=SAMPLE_RATE, frame_ms=30, hop_ms=10):
    """Detect stable speech via F0 estimation (autocorrelation)."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    min_lag = int(sr / 300)   # 300Hz upper bound
    max_lag = int(sr / 85)    # 85Hz lower bound

    voiced_in_range = 0
    voiced_total = 0

    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start:start + frame_len]
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        if corr[0] > 0:
            corr = corr / corr[0]
        search = corr[min_lag:max_lag + 1]
        if len(search) == 0:
            continue
        if np.max(search) > 0.3:
            voiced_total += 1
            peak_lag = np.argmax(search) + min_lag
            f0 = sr / peak_lag
            if 85 <= f0 <= 300:
                voiced_in_range += 1

    if voiced_total == 0:
        return False
    return (voiced_in_range / voiced_total) >= 0.50


def load_thresholds(thresholds_path):
    if thresholds_path and os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            t = json.load(f)
        amber = t.get('youden', 0.50)
        red = t.get('low_fpr', 0.70)
        return amber, red, thresholds_path
    return 0.50, 0.70, None


def main():
    parser = argparse.ArgumentParser(description='SafeCommute AI — Live Demo')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--thresholds', type=str, default=None)
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()

    # Auto-detect threshold file
    if args.thresholds is None and args.model.startswith('models/'):
        env = os.path.basename(args.model).replace('_model.pth', '')
        candidate = f'models/{env}_thresholds.json'
        if os.path.exists(candidate):
            args.thresholds = candidate

    if not os.path.exists(args.model):
        print(f"Error: {args.model} not found.")
        sys.exit(1)

    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()

    mean, std = 0.0, 1.0
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            s = json.load(f)
        mean, std = s['mean'], s['std']

    amber, red, thresh_source = load_thresholds(args.thresholds)

    p = pyaudio.PyAudio()
    try:
        info = p.get_default_input_device_info()
        mic_name = info['name']
    except Exception:
        print("Error: No microphone found.")
        p.terminate()
        sys.exit(1)

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                    input=True, frames_per_buffer=CHUNK)

    print(f"\n{'='*58}")
    print(f"  SafeCommute AI — Live Demo ({args.duration}s)")
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Mic: {mic_name}")
    print(f"  Features: PCEN (gain-invariant, no mic calibration needed)")
    if thresh_source:
        print(f"  Thresholds: amber={amber:.3f}, red={red:.3f} (from {thresh_source})")
    else:
        print(f"  Thresholds: amber={amber:.2f}, red={red:.2f} (default)")
    print(f"  Privacy: RAM only, no audio saved")
    print(f"  Speech-aware: threshold -> {SPEECH_THRESH_BOOST:.2f} during speech")
    print(f"{'='*58}\n")

    audio_buf = np.zeros(BUFFER, dtype=np.float32)
    history = collections.deque(maxlen=SMOOTHING)
    start_time = time.time()
    last_status = None

    # ── Auto-calibrate: measure ambient baseline for 3 seconds ────────
    print("  Calibrating ambient baseline (3s — stay quiet)...")
    cal_probs = []
    for _ in range(3):
        data = stream.read(CHUNK, exception_on_overflow=False)
        new = np.frombuffer(data, dtype=np.float32)
        audio_buf = np.roll(audio_buf, -CHUNK)
        audio_buf[-CHUNK:] = new
        rms = float(np.sqrt(np.mean(audio_buf ** 2)))
        if rms >= ENERGY_GATE_RMS:
            feat = preprocess(audio_buf, mean, std)
            with torch.no_grad():
                p = torch.softmax(model(feat), dim=1)[0][1].item()
            cal_probs.append(p)
    baseline_prob = float(np.mean(cal_probs)) if cal_probs else 0.0
    print(f"  Baseline: {baseline_prob:.3f} (will be subtracted from raw output)\n")

    try:
        while time.time() - start_time < args.duration:
            data = stream.read(CHUNK, exception_on_overflow=False)
            new = np.frombuffer(data, dtype=np.float32)
            audio_buf = np.roll(audio_buf, -CHUNK)
            audio_buf[-CHUNK:] = new

            rms = float(np.sqrt(np.mean(audio_buf ** 2)))
            remaining = max(0, args.duration - (time.time() - start_time))
            t = time.strftime('%H:%M:%S')

            # Energy gating on raw audio — silence = auto-safe
            if rms < ENERGY_GATE_RMS:
                print(f"  [{t}]  SAFE    [--------------------] 0.00 (silent)  [{remaining:.0f}s]", end='\r')
                history.clear()
                last_status = 'silent'
                continue

            # Speech detection: raise threshold during normal conversation
            is_speech = detect_speech(audio_buf)

            # PCEN handles gain normalization — no mic_scale needed
            feat = preprocess(audio_buf, mean, std)
            with torch.no_grad():
                raw_prob = torch.softmax(model(feat), dim=1)[0][1].item()

            # Subtract ambient baseline so normal conditions sit near 0.0
            prob = max(0.0, (raw_prob - baseline_prob) / max(1.0 - baseline_prob, 0.01))

            history.append(prob)
            smoothed = float(np.mean(history))

            # Adaptive thresholding
            eff_amber = (SPEECH_THRESH_BOOST - 0.10) if is_speech else amber
            eff_red = SPEECH_THRESH_BOOST if is_speech else red
            speech_tag = "SPK" if is_speech else "   "

            bar = '#' * int(smoothed * 20) + '-' * (20 - int(smoothed * 20))

            if smoothed >= eff_red:
                if last_status != 'red':
                    print()
                print(f"  [{t}]  ALERT   [{bar}] {smoothed:.2f} (raw={prob:.2f}) {speech_tag}  ESCALATION")
                last_status = 'red'
            elif smoothed >= eff_amber:
                print(f"  [{t}]  WARNING [{bar}] {smoothed:.2f} (raw={prob:.2f}) {speech_tag}  [{remaining:.0f}s]", end='\r')
                last_status = 'amber'
            else:
                print(f"  [{t}]  SAFE    [{bar}] {smoothed:.2f} (raw={prob:.2f}) {speech_tag}  [{remaining:.0f}s]", end='\r')
                if last_status == 'red':
                    print()
                last_status = 'safe'

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f"\n\nDone. No audio was stored.")


if __name__ == '__main__':
    main()
