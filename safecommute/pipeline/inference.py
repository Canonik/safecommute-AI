import collections
import json
import os
import sys
import time

import numpy as np
import pyaudio
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.features import preprocess
from safecommute.constants import (
    SAMPLE_RATE, N_MELS, TIME_FRAMES,
    MODEL_SAVE_PATH as MODEL_PATH,
    STATS_PATH, THRESHOLDS_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_WINDOW_SEC = 3
STRIDE_SEC         = 1
CHUNK_SIZE         = int(SAMPLE_RATE * STRIDE_SEC)
BUFFER_SIZE        = int(SAMPLE_RATE * CONTEXT_WINDOW_SEC)
SMOOTHING_WINDOW   = 4
ENERGY_GATE_RMS    = 0.003
PREFERRED_MIC      = None


# ─────────────────────────────────────────────────────────────────────────────
# MICROPHONE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def list_input_devices(p):
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}  "
                  f"(ch={info['maxInputChannels']}, sr={int(info['defaultSampleRate'])})")


def find_input_device(p, preferred=None):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] < 1:
            continue
        if preferred and preferred.lower() in info['name'].lower():
            print(f"  Selected preferred mic: [{i}] {info['name']}")
            return i
    try:
        d = p.get_default_input_device_info()
        print(f"  Using default mic: [{d['index']}] {d['name']}")
        return d['index']
    except IOError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load stats ────────────────────────────────────────────────────────
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            s = json.load(f)
        feat_mean, feat_std = s['mean'], s['std']
        print(f"Feature stats: mean={feat_mean:.2f}, std={feat_std:.2f}")
    else:
        print(f"Warning: '{STATS_PATH}' not found — normalisation disabled.")
        feat_mean, feat_std = 0.0, 1.0

    # ── Load thresholds ───────────────────────────────────────────────────
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            t = json.load(f)
        amber_thresh = t.get('amber', 0.40)
        red_thresh   = t.get('red',   0.70)
        print(f"Thresholds loaded: amber={amber_thresh}, red={red_thresh}")
    else:
        amber_thresh, red_thresh = 0.40, 0.70
        print("thresholds.json not found — using defaults (run train_model.py to calibrate).")

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run train_model.py first.")
        sys.exit(1)
    device = torch.device("cpu")
    model  = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.\n")

    # ── Init PyAudio ──────────────────────────────────────────────────────
    p = pyaudio.PyAudio()
    list_input_devices(p)
    device_idx = find_input_device(p, preferred=PREFERRED_MIC)
    if device_idx is None:
        print("Error: No input device found.")
        p.terminate()
        sys.exit(1)

    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=CHUNK_SIZE,
        )
    except Exception as e:
        print(f"Microphone error: {e}")
        p.terminate()
        sys.exit(1)

    print("=" * 58)
    print("  🎙️  SAFECOMMUTE AI  —  LIVE INFERENCE ACTIVE")
    print("  🔒  GDPR mode: RAM only. No audio recorded.")
    print(f"  📊  Smoothing: {SMOOTHING_WINDOW} strides | "
          f"VAD gate RMS: {ENERGY_GATE_RMS}")
    print(f"  🟠  Amber ≥ {amber_thresh:.2f}  |  🔴  Red ≥ {red_thresh:.2f}")
    print("=" * 58 + "\n")

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    prob_history = collections.deque(maxlen=SMOOTHING_WINDOW)
    last_status  = None

    try:
        while True:
            data      = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            new_audio = np.frombuffer(data, dtype=np.float32)

            audio_buffer        = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = new_audio

            rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
            if rms < ENERGY_GATE_RMS:
                ts = time.strftime('%H:%M:%S')
                print(f"[{ts}] 🔵 SILENT  (RMS={rms:.4f}, VAD gate)          ", end='\r')
                continue

            features = preprocess(audio_buffer, feat_mean, feat_std).to(device)
            with torch.no_grad():
                logits     = model(features)
                raw_unsafe = torch.softmax(logits, dim=1)[0][1].item()

            prob_history.append(raw_unsafe)
            smoothed = float(np.mean(prob_history))

            ts  = time.strftime('%H:%M:%S')
            bar = "█" * int(smoothed * 20) + "░" * (20 - int(smoothed * 20))

            if smoothed >= red_thresh:
                status = "🔴 ALERT  "
            elif smoothed >= amber_thresh:
                status = "🟠 WARNING"
            else:
                status = "🟢 SAFE   "

            line = (f"[{ts}] {status} [{bar}] {smoothed:.2f} "
                    f"(raw={raw_unsafe:.2f}, RMS={rms:.3f})")

            if smoothed >= red_thresh:
                print(f"\n{'!'*58}")
                print(f"  {line}  ⚠️  ESCALATION DETECTED!")
                print(f"{'!'*58}")
                last_status = 'red'
            else:
                print(line + "          ", end='\r', flush=True)
                if last_status == 'red':
                    print()
                last_status = 'amber' if smoothed >= amber_thresh else 'green'

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping SafeCommute AI…")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("✅ Shut down cleanly. No data was stored.")


if __name__ == "__main__":
    main()
