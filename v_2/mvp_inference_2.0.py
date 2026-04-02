import json
import os
import sys
import time
import collections

import numpy as np
import librosa
import pyaudio
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000
CONTEXT_WINDOW_SEC  = 3
STRIDE_SEC          = 1
CHUNK_SIZE          = int(SAMPLE_RATE * STRIDE_SEC)
BUFFER_SIZE         = int(SAMPLE_RATE * CONTEXT_WINDOW_SEC)
N_MELS              = 64
MODEL_PATH          = "safecommute_edge_model.pth"
STATS_PATH          = "feature_stats.json"

# Smoothing: keep a rolling window of the last N predictions.
# RED fires only when the AVERAGE probability over this window exceeds the threshold.
# This kills single-frame false positives while keeping real escalations.
SMOOTHING_WINDOW    = 4    # ~4 seconds of context for a final decision
AMBER_THRESHOLD     = 0.40
RED_THRESHOLD       = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (must exactly mirror train_model.py)
# ─────────────────────────────────────────────────────────────────────────────
class SafeCommuteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v2(weights=None)
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# (must exactly mirror data_pipeline.py — especially ref=1.0)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(audio_buffer, mean, std):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=1024, hop_length=256           # same as pipeline
    )
    # CRITICAL: ref=1.0 (fixed), not ref=np.max (per-clip normalisation)
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)

    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Apply the same z-score normalisation used during training
    tensor = (tensor - mean) / (std + 1e-8)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# MICROPHONE AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_input_device(p, preferred_name_fragment=None):
    """
    Return the index of the best available input device.
    If preferred_name_fragment is provided, prefer a device whose name
    contains that string (case-insensitive). Falls back to PyAudio default.
    """
    best_idx  = None
    best_ch   = 0

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] < 1:
            continue
        name = info['name'].lower()
        # Prefer explicitly named device
        if preferred_name_fragment and preferred_name_fragment.lower() in name:
            print(f"  Selected preferred mic: [{i}] {info['name']}")
            return i
        # Otherwise track the device with the most input channels as a heuristic
        if info['maxInputChannels'] > best_ch:
            best_ch  = info['maxInputChannels']
            best_idx = i

    # Fall back to PyAudio's system default
    try:
        default = p.get_default_input_device_info()
        print(f"  Using default mic: [{default['index']}] {default['name']}")
        return default['index']
    except IOError:
        if best_idx is not None:
            print(f"  Falling back to best available mic: [{best_idx}]")
            return best_idx
        return None


def list_input_devices(p):
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}  (ch={info['maxInputChannels']},"
                  f" sr={int(info['defaultSampleRate'])})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load normalisation stats ──────────────────────────────────────────
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            stats = json.load(f)
        feat_mean = stats['mean']
        feat_std  = stats['std']
        print(f"Loaded feature stats: mean={feat_mean:.2f}, std={feat_std:.2f}")
    else:
        print(f"Warning: '{STATS_PATH}' not found. Normalisation disabled.")
        print("  → Re-run data_pipeline.py to fix this.")
        feat_mean, feat_std = 0.0, 1.0

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading SafeCommute AI edge model…")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run train_model.py first.")
        sys.exit(1)

    device = torch.device("cpu")   # edge inference: CPU avoids latency spikes
    model  = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # ── Init PyAudio ──────────────────────────────────────────────────────
    p = pyaudio.PyAudio()
    list_input_devices(p)

    # Pass a fragment of your mic name to prefer it, e.g. "pipewire" or "usb"
    # Leave None to auto-detect.
    preferred_mic = None
    device_idx    = find_input_device(p, preferred_name_fragment=preferred_mic)

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

    # ── Inference loop ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  🎙️  SAFECOMMUTE AI  —  LIVE INFERENCE ACTIVE")
    print("  🔒  GDPR mode: RAM only. No audio stored.")
    print(f"  📊  Smoothing window: {SMOOTHING_WINDOW} strides (~{SMOOTHING_WINDOW}s)")
    print("=" * 55 + "\n")

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    # Rolling window of raw unsafe probabilities for smoothing
    prob_history = collections.deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            # 1. Read one stride of live audio
            data      = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            new_audio = np.frombuffer(data, dtype=np.float32)

            # 2. Slide the rolling buffer
            audio_buffer        = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = new_audio

            # 3. Feature extraction + normalisation
            features = preprocess(audio_buffer, feat_mean, feat_std).to(device)

            # 4. Inference
            with torch.no_grad():
                logits      = model(features)
                probs       = torch.softmax(logits, dim=1)
                raw_unsafe  = probs[0][1].item()

            # 5. Smooth probabilities over the rolling window
            prob_history.append(raw_unsafe)
            smoothed_unsafe = float(np.mean(prob_history))

            # 6. Display
            ts = time.strftime('%H:%M:%S')
            if smoothed_unsafe < AMBER_THRESHOLD:
                status = "🟢 SAFE    "
            elif smoothed_unsafe < RED_THRESHOLD:
                status = "🟠 WARNING "
            else:
                status = "🔴 ALERT   "

            bar_filled = int(smoothed_unsafe * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)

            print(
                f"[{ts}] {status} | [{bar}] {smoothed_unsafe:.2f} "
                f"(raw={raw_unsafe:.2f})"
                + ("  ⚠️  ESCALATION DETECTED!" if smoothed_unsafe >= RED_THRESHOLD else ""),
                end='\r',
                flush=True,
            )
            # Print newline on state changes so history is readable
            if smoothed_unsafe >= RED_THRESHOLD:
                print()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping SafeCommute AI…")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("✅ Shut down cleanly. No data was stored.")


if __name__ == "__main__":
    main()