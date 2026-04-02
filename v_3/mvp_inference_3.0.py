import collections
import json
import os
import sys
import time

import numpy as np
import librosa
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
CONTEXT_WINDOW_SEC = 3
STRIDE_SEC         = 1
CHUNK_SIZE         = int(SAMPLE_RATE * STRIDE_SEC)
BUFFER_SIZE        = int(SAMPLE_RATE * CONTEXT_WINDOW_SEC)
N_MELS             = 64
TIME_FRAMES        = 188    # ceil(48000 / 256) — must match training
MODEL_PATH         = "safecommute_edge_model.pth"
STATS_PATH         = "feature_stats.json"
THRESHOLDS_PATH    = "thresholds.json"

# Smoothing: average the last N probability values before thresholding.
# Increase to reduce sensitivity; decrease for faster response.
SMOOTHING_WINDOW   = 4

# Energy VAD gate: skip model inference when the buffer is too quiet.
# This prevents silent frames or pure HVAC hum from needlessly burning CPU
# and occasionally generating junk probabilities.
# Unit: RMS amplitude of the normalised float32 audio buffer.
# A typical spoken voice is 0.01–0.1; silence / very quiet noise is < 0.002.
ENERGY_GATE_RMS    = 0.003

# Preferred microphone name fragment (case-insensitive).
# Set to None for auto-detect, or e.g. "pipewire" / "usb" / "blue yeti".
PREFERRED_MIC      = None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — must exactly mirror train_model.py
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, pool=(2, 2)):
        return F.avg_pool2d(self.net(x), pool)


class SafeCommuteCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.block1   = ConvBlock(1,   64)
        self.block2   = ConvBlock(64,  128)
        self.block3   = ConvBlock(128, 256)

        freq_after_blocks = n_mels // (2 ** 3)   # 64 // 8 = 8

        self.freq_reduce = nn.Linear(256 * freq_after_blocks, 256)
        self.gru         = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        B, C, Freq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Freq)
        x = F.relu(self.freq_reduce(x))
        _, h = self.gru(x)
        x = self.dropout(h.squeeze(0))
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — must exactly mirror data_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(audio_buffer, mean, std):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=1024, hop_length=256
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)   # fixed reference
    tensor  = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Enforce TIME_FRAMES (same as dataset loader)
    t = tensor.shape[-1]
    if t > TIME_FRAMES:
        tensor = tensor[:, :, :, :TIME_FRAMES]
    elif t < TIME_FRAMES:
        pad    = torch.zeros(1, 1, N_MELS, TIME_FRAMES - t)
        tensor = torch.cat([tensor, pad], dim=-1)

    return (tensor - mean) / (std + 1e-8)


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
            # 1. Read one stride
            data      = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            new_audio = np.frombuffer(data, dtype=np.float32)

            # 2. Slide rolling buffer
            audio_buffer        = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = new_audio

            # 3. Energy VAD gate
            rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
            if rms < ENERGY_GATE_RMS:
                # Buffer is silent / near-silent — skip inference entirely.
                # Keep the probability history stale rather than injecting 0.0,
                # so the display doesn't flicker when someone stops talking briefly.
                ts  = time.strftime('%H:%M:%S')
                print(f"[{ts}] 🔵 SILENT  (RMS={rms:.4f}, VAD gate)          ", end='\r')
                continue

            # 4. Feature extraction + inference
            features    = preprocess(audio_buffer, feat_mean, feat_std).to(device)
            with torch.no_grad():
                logits      = model(features)
                raw_unsafe  = torch.softmax(logits, dim=1)[0][1].item()

            # 5. Temporal smoothing
            prob_history.append(raw_unsafe)
            smoothed = float(np.mean(prob_history))

            # 6. Display
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
                    print()  # newline after red clears
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