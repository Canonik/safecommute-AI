"""
Shared constants for SafeCommute AI.
All scripts must use these values to stay in sync.
"""

# Audio
SAMPLE_RATE = 16000
DURATION_SEC = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)  # 48000

# Spectrogram
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TIME_FRAMES = 188  # ceil(48000 / 256)

# Paths
DATA_DIR = "prepared_data"
RAW_DIR = "raw_data"
STATS_PATH = "feature_stats.json"
THRESHOLDS_PATH = "thresholds.json"
MODEL_SAVE_PATH = "safecommute_edge_model.pth"

# Reproducibility
SEED = 42
