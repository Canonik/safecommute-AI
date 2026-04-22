"""
Real-time inference engine for SafeCommute AI.

Captures live audio from a microphone, runs the SafeCommuteCNN model on
sliding 3-second windows, and displays a traffic-light status (green/amber/red).

Privacy-preserving design (GDPR-compliant):
  - Audio is kept in a RAM-only rolling buffer — never written to disk.
  - Only non-reconstructible mel spectrograms are fed to the model.
  - No raw audio, no spectrograms, and no model outputs are persisted.

Signal processing pipeline:
  1. Audio capture: 1-second chunks from PyAudio (16 kHz, float32, mono).
  2. Rolling buffer: 3-second window updated by shifting 1 second at a time.
     This gives 1-second response granularity with 3-second context.
  3. Energy gating: if RMS < 0.003, skip inference entirely (auto-safe).
     This avoids wasting CPU on silence and prevents the model from
     producing spurious outputs on near-zero input (the model was not
     trained on true digital silence and may produce arbitrary outputs).
  4. Feature extraction: log-mel spectrogram via preprocess() from features.py.
  5. Model inference: SafeCommuteCNN outputs softmax probability of "unsafe".
  6. Temporal smoothing: moving average over SMOOTHING_WINDOW (4) consecutive
     predictions. This reduces single-frame false positives — a brief cough
     might spike one window but won't sustain across 4 seconds.
  6b. Temporal-majority aggregation (MAJORITY_K): independent of (6), a red
     alert fires only when the RAW (un-smoothed) probability has been over
     the red threshold for MAJORITY_K consecutive strides. Any sub-threshold
     stride resets the counter. Smoother = amplitude criterion, majority gate
     = duration criterion — together they suppress single-window spikes from
     speech/crowd that dominated the metro deployment-gap tweak plateau
     (paper.md §1.4, §7). Set MAJORITY_K=1 to disable.
  7. Dual thresholds: amber (warning) and red (alert) levels.
     - Below amber: green (safe).
     - Between amber and red: amber (elevated, monitoring).
     - Above red: red (alert, potential escalation detected).
     The dual-threshold design avoids binary flip-flopping and gives
     operators time to assess before a full alert triggers.

Usage:
    PYTHONPATH=. python safecommute/pipeline/inference.py
"""

import collections
import json
import os
import sys
import time

import numpy as np
import pyaudio
import torch

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
CONTEXT_WINDOW_SEC = 3       # Model input window — matches training duration
STRIDE_SEC         = 1       # How often we run inference (response granularity)
CHUNK_SIZE         = int(SAMPLE_RATE * STRIDE_SEC)     # 16000 samples per read
BUFFER_SIZE        = int(SAMPLE_RATE * CONTEXT_WINDOW_SEC)  # 48000 rolling buffer
SMOOTHING_WINDOW   = 4       # Moving average over 4 predictions (~4 seconds)
ENERGY_GATE_RMS    = 0.003   # Below this RMS, audio is considered silence
# Temporal-majority aggregation: alert only when the *raw* (un-smoothed)
# per-stride probability has been >= the effective red threshold for
# MAJORITY_K consecutive strides. Complements — does not replace — the
# 4-frame smoother. The smoother attenuates amplitude; the majority gate
# enforces duration. Together they suppress single-window spikes from
# speech / metal / crowd that dominated the architecture-preserving tweak
# plateau (paper.md §1.4, §7). Set MAJORITY_K=1 to disable (pre-gate behaviour).
MAJORITY_K         = 2
PREFERRED_MIC      = None    # Set to substring of mic name to auto-select

# Speech-aware thresholding: when stable speech is detected, raise the
# alert threshold to avoid false-positiving on normal conversation.
# Normal speech has stable F0 between 85-300Hz. Screams have F0 > 500Hz
# or wildly unstable pitch. This separation is robust across languages.
SPEECH_THRESH_BOOST = 0.70   # Threshold during detected speech
F0_MIN_HZ          = 85      # Lowest expected speech F0
F0_MAX_HZ          = 300     # Highest expected speech F0
F0_STABILITY_RATIO  = 0.50   # Fraction of frames that must have stable F0


# ─────────────────────────────────────────────────────────────────────────────
# SPEECH DETECTION (pitch-based)
# ─────────────────────────────────────────────────────────────────────────────
def detect_speech(audio, sr=SAMPLE_RATE, frame_ms=30, hop_ms=10):
    """
    Detect stable speech by estimating fundamental frequency (F0) via
    autocorrelation. Returns True if a majority of voiced frames have
    F0 in the normal speech range (85-300Hz).

    Normal speech: stable F0 in [85, 300] Hz across most voiced frames.
    Screams/yells: F0 > 500Hz, or rapidly changing, or no clear F0.
    Gunshots/glass: no periodic structure, F0 estimation fails.

    This runs in <1ms on a 3-second buffer — negligible overhead.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    # Lag bounds for F0 range: lag = sr / f0
    min_lag = int(sr / F0_MAX_HZ)  # ~53 samples for 300Hz
    max_lag = int(sr / F0_MIN_HZ)  # ~188 samples for 85Hz

    voiced_in_range = 0
    voiced_total = 0

    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start:start + frame_len]

        # Skip quiet frames
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            continue

        # Autocorrelation-based F0 estimation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]  # keep positive lags only

        # Normalize
        if corr[0] > 0:
            corr = corr / corr[0]

        # Search for peak in the speech F0 lag range
        search = corr[min_lag:max_lag + 1]
        if len(search) == 0:
            continue

        peak_val = np.max(search)

        # A clear periodic signal has autocorrelation peak > 0.3
        if peak_val > 0.3:
            voiced_total += 1
            peak_lag = np.argmax(search) + min_lag
            f0 = sr / peak_lag
            if F0_MIN_HZ <= f0 <= F0_MAX_HZ:
                voiced_in_range += 1

    if voiced_total == 0:
        return False, 0.0

    ratio = voiced_in_range / voiced_total
    avg_f0 = 0.0  # could compute if needed

    return ratio >= F0_STABILITY_RATIO, ratio


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
    print("  SAFECOMMUTE AI  —  LIVE INFERENCE ACTIVE")
    print("  GDPR mode: RAM only. No audio recorded.")
    print(f"  Smoothing: {SMOOTHING_WINDOW} strides | "
          f"VAD gate RMS: {ENERGY_GATE_RMS} | "
          f"Majority-k: {MAJORITY_K}")
    print(f"  Amber >= {amber_thresh:.2f}  |  Red >= {red_thresh:.2f}")
    print(f"  Speech-aware: threshold -> {SPEECH_THRESH_BOOST:.2f} "
          f"when speech detected")
    print("=" * 58 + "\n")

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    prob_history = collections.deque(maxlen=SMOOTHING_WINDOW)
    consecutive_over = 0   # counts consecutive strides with raw_unsafe >= red
    last_status  = None

    # ── Auto-calibrate: measure ambient baseline for 3 seconds ────────
    print("Calibrating ambient baseline (3s — stay quiet)...")
    cal_probs = []
    for _ in range(3):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        new_audio = np.frombuffer(data, dtype=np.float32)
        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
        audio_buffer[-CHUNK_SIZE:] = new_audio
        rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
        if rms >= ENERGY_GATE_RMS:
            features = preprocess(audio_buffer, feat_mean, feat_std).to(device)
            with torch.no_grad():
                p = torch.softmax(model(features), dim=1)[0][1].item()
            cal_probs.append(p)
    baseline_prob = float(np.mean(cal_probs)) if cal_probs else 0.0
    print(f"Baseline: {baseline_prob:.3f} (subtracted from raw output)\n")

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

            # Speech detection: check if audio contains stable speech F0.
            # If yes, raise alert threshold to avoid false-positiving on
            # normal conversation. Screams/yells will still trigger because
            # they have completely different pitch characteristics.
            is_speech, speech_ratio = detect_speech(audio_buffer)

            features = preprocess(audio_buffer, feat_mean, feat_std).to(device)
            with torch.no_grad():
                logits     = model(features)
                raw_unsafe = torch.softmax(logits, dim=1)[0][1].item()

            # Gentle baseline correction — subtract half the ambient floor
            adjusted = max(0.0, raw_unsafe - baseline_prob * 0.5)

            prob_history.append(adjusted)
            smoothed = float(np.mean(prob_history))

            # Adaptive thresholding: during speech, require higher confidence
            if is_speech:
                eff_amber = SPEECH_THRESH_BOOST - 0.10  # 0.75
                eff_red   = SPEECH_THRESH_BOOST          # 0.85
            else:
                eff_amber = amber_thresh
                eff_red   = red_thresh

            ts  = time.strftime('%H:%M:%S')
            bar = "█" * int(smoothed * 20) + "░" * (20 - int(smoothed * 20))
            speech_tag = "🗣️" if is_speech else "  "

            # Temporal-majority gate: count consecutive strides whose RAW
            # (pre-smoothing) prob is >= red. We alert only when (smoothed
            # >= red) AND (consecutive_over >= MAJORITY_K). The smoothed
            # check is the amplitude criterion; the counter is the duration
            # criterion. Resetting on any sub-threshold stride enforces
            # strict consecutiveness — a single quiet window clears the gate
            # and starts it over.
            if raw_unsafe >= eff_red:
                consecutive_over += 1
            else:
                consecutive_over = 0

            smoothed_red = smoothed >= eff_red
            majority_ok = consecutive_over >= MAJORITY_K
            fire_red = smoothed_red and majority_ok

            if fire_red:
                status = "🔴 ALERT  "
            elif smoothed >= eff_amber:
                status = "🟠 WARNING"
            else:
                status = "🟢 SAFE   "

            line = (f"[{ts}] {status} {speech_tag} [{bar}] {smoothed:.2f} "
                    f"(raw={raw_unsafe:.2f}, "
                    f"k={consecutive_over}/{MAJORITY_K}, "
                    f"RMS={rms:.3f})")

            if fire_red:
                print(f"\n{'!'*58}")
                print(f"  {line}  ESCALATION DETECTED!")
                print(f"{'!'*58}")
                last_status = 'red'
            else:
                print(line + "          ", end='\r', flush=True)
                if last_status == 'red':
                    print()
                last_status = 'amber' if smoothed >= eff_amber else 'green'

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping SafeCommute AI…")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("✅ Shut down cleanly. No data was stored.")


if __name__ == "__main__":
    main()
