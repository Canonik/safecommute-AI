"""
Shared audio and feature constants for the PCEN reconstruction audit.

The privacy harness and retained classifier case-study code use the same
3-second, 16 kHz, 64-mel frontend. These constants are not a privacy
guarantee; they simply define the representation under attack.
"""

# Audio: 16 kHz mono is standard for speech/environmental audio tasks.
# 3-second windows capture the typical duration of a scream or shout while
# matching the feature tile evaluated by the reconstruction attack.
SAMPLE_RATE = 16000
DURATION_SEC = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)  # 48000

# Spectrogram: these settings yield a (64, 188) mel spectrogram per window.
# N_FFT=1024 gives ~64 ms analysis frames at 16 kHz (good frequency resolution).
# HOP_LENGTH=256 gives ~16 ms hop (good time resolution for transient events
# like glass breaking). TIME_FRAMES = ceil(48000 / 256) = 188.
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TIME_FRAMES = 188  # ceil(48000 / 256)

# Optional local paths used by retained classifier audit helpers. The public
# release intentionally does not include these generated artifacts.
DATA_DIR = "prepared_data"
RAW_DIR = "raw_data"
STATS_PATH = "feature_stats.json"
THRESHOLDS_PATH = "thresholds.json"
MODEL_SAVE_PATH = "models/safecommute_v2.pth"

# Reproducibility
SEED = 42
