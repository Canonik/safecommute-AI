#!/usr/bin/env python3
"""
SafeCommute AI — Quick Demo

Records 10 seconds from your microphone and classifies each 3-second
window as Safe/Warning/Alert. No audio is saved to disk.

Usage:
    PYTHONPATH=. python demo.py
"""

import sys
import os
import time
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import pyaudio
from safecommute.model import SafeCommuteCNN
from safecommute.features import preprocess
from safecommute.constants import SAMPLE_RATE, MODEL_SAVE_PATH, STATS_PATH

# Config
DURATION = 50          # seconds to record
WINDOW_SEC = 3
STRIDE_SEC = 1
CHUNK = int(SAMPLE_RATE * STRIDE_SEC)
BUFFER = int(SAMPLE_RATE * WINDOW_SEC)
SMOOTHING = 3

def main():
    # Load model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: {MODEL_SAVE_PATH} not found. Train first.")
        sys.exit(1)

    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=True))
    model.eval()

    # Load stats
    mean, std = 0.0, 1.0
    if os.path.exists(STATS_PATH):
        import json
        with open(STATS_PATH) as f:
            s = json.load(f)
        mean, std = s['mean'], s['std']

    # Open mic
    p = pyaudio.PyAudio()
    try:
        info = p.get_default_input_device_info()
        print(f"Mic: {info['name']}")
    except:
        print("Error: No microphone found.")
        p.terminate()
        sys.exit(1)

    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                    input=True, frames_per_buffer=CHUNK)

    print(f"\n{'='*50}")
    print(f"  SafeCommute AI — Live Demo ({DURATION}s)")
    print(f"  Model: 1.83M params, 7MB, ~9ms inference")
    print(f"  Privacy: RAM only, no audio saved")
    print(f"{'='*50}\n")

    audio_buf = np.zeros(BUFFER, dtype=np.float32)
    history = collections.deque(maxlen=SMOOTHING)
    start = time.time()

    try:
        while time.time() - start < DURATION:
            data = stream.read(CHUNK, exception_on_overflow=False)
            new = np.frombuffer(data, dtype=np.float32)
            audio_buf = np.roll(audio_buf, -CHUNK)
            audio_buf[-CHUNK:] = new

            rms = float(np.sqrt(np.mean(audio_buf ** 2)))
            if rms < 0.003:
                t = time.strftime('%H:%M:%S')
                remaining = max(0, DURATION - (time.time() - start))
                print(f"  [{t}]  SILENT  (RMS={rms:.4f})  [{remaining:.0f}s left]", end='\r')
                continue

            feat = preprocess(audio_buf, mean, std)
            with torch.no_grad():
                prob = torch.softmax(model(feat), dim=1)[0][1].item()

            history.append(prob)
            smoothed = float(np.mean(history))

            t = time.strftime('%H:%M:%S')
            bar = '█' * int(smoothed * 20) + '░' * (20 - int(smoothed * 20))
            remaining = max(0, DURATION - (time.time() - start))

            if smoothed >= 0.7:
                print(f"\n  [{t}]  🔴 ALERT   [{bar}] {smoothed:.2f}  ⚠️  ESCALATION")
            elif smoothed >= 0.4:
                print(f"  [{t}]  🟠 WARNING [{bar}] {smoothed:.2f}  [{remaining:.0f}s]", end='\r')
            else:
                print(f"  [{t}]  🟢 SAFE    [{bar}] {smoothed:.2f}  [{remaining:.0f}s]", end='\r')

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f"\n\nDone. No audio was stored.")


if __name__ == '__main__':
    main()
