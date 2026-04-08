#!/usr/bin/env python3
"""
SafeCommute AI — Live Demo

Records from your microphone and classifies each 3-second window as
Safe/Warning/Alert in real-time. No audio is saved to disk.

Usage:
    PYTHONPATH=. python demo.py
    PYTHONPATH=. python demo.py --model models/cycle6_noise_inject.pth
    PYTHONPATH=. python demo.py --threshold 0.60
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
SMOOTHING = 3          # shorter smoothing = faster response
ENERGY_GATE_RMS = 0.005


def main():
    parser = argparse.ArgumentParser(description='SafeCommute AI — Live Demo')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Alert threshold (default 0.55, lower = more sensitive)')
    parser.add_argument('--duration', type=int, default=120)
    args = parser.parse_args()

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

    red = args.threshold
    amber = red - 0.15

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
    print(f"  Alert >= {red:.2f} | Warning >= {amber:.2f}")
    print(f"  Privacy: RAM only, no audio saved")
    print(f"{'='*58}\n")

    audio_buf = np.zeros(BUFFER, dtype=np.float32)
    history = collections.deque(maxlen=SMOOTHING)
    last_status = None

    try:
        while time.time() - time.time() < args.duration or True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            new = np.frombuffer(data, dtype=np.float32)
            audio_buf = np.roll(audio_buf, -CHUNK)
            audio_buf[-CHUNK:] = new

            rms = float(np.sqrt(np.mean(audio_buf ** 2)))
            t = time.strftime('%H:%M:%S')

            if rms < ENERGY_GATE_RMS:
                print(f"  [{t}]  SAFE    [--------------------] 0.00 (silent)     ", end='\r')
                history.clear()
                last_status = 'silent'
                continue

            feat = preprocess(audio_buf, mean, std)
            with torch.no_grad():
                prob = torch.softmax(model(feat), dim=1)[0][1].item()

            history.append(prob)
            smoothed = float(np.mean(history))

            bar = '#' * int(smoothed * 20) + '-' * (20 - int(smoothed * 20))

            if smoothed >= red:
                if last_status != 'red':
                    print()
                print(f"  [{t}]  ALERT   [{bar}] {smoothed:.2f} (raw={prob:.2f}, rms={rms:.3f})")
                last_status = 'red'
            elif smoothed >= amber:
                print(f"  [{t}]  WARNING [{bar}] {smoothed:.2f} (raw={prob:.2f}, rms={rms:.3f})   ", end='\r')
                last_status = 'amber'
            else:
                print(f"  [{t}]  SAFE    [{bar}] {smoothed:.2f} (raw={prob:.2f}, rms={rms:.3f})   ", end='\r')
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
