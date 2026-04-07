"""
Shared constants for SafeCommute AI.

Single source of truth for all hyperparameters and paths. Every pipeline
script, feature extractor, and inference module imports from here to ensure
consistency. Changing a value here propagates everywhere — but note that
existing .pt files will NOT be re-generated automatically.

These values were chosen for the edge deployment target:
  - 3-second window balances temporal context vs. latency (shorter windows
    miss escalation patterns; longer windows increase response delay).
  - 16 kHz sample rate retains speech and threat frequency content while
    halving data vs. 32 kHz — most threat-relevant energy is below 8 kHz.
  - 64 mel bins provide sufficient frequency resolution for distinguishing
    screams from laughter/cheering without bloating the model input.
"""

# Audio — 16 kHz mono is standard for speech/environmental audio tasks.
# 3-second windows capture the typical duration of a scream or shout while
# keeping inference latency under 10 ms on CPU.
SAMPLE_RATE = 16000
DURATION_SEC = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)  # 48000

# Spectrogram — these settings yield a (64, 188) mel spectrogram per window.
# N_FFT=1024 gives ~64 ms analysis frames at 16 kHz (good frequency resolution).
# HOP_LENGTH=256 gives ~16 ms hop (good time resolution for transient events
# like glass breaking). TIME_FRAMES = ceil(48000 / 256) = 188.
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TIME_FRAMES = 188  # ceil(48000 / 256)

# Paths — all relative to repo root (scripts must run with PYTHONPATH=.)
DATA_DIR = "prepared_data"
RAW_DIR = "raw_data"
STATS_PATH = "feature_stats.json"
THRESHOLDS_PATH = "thresholds.json"
MODEL_SAVE_PATH = "safecommute_edge_model.pth"

# Reproducibility
SEED = 42
