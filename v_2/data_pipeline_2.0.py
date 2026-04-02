import os
import json
import requests
import zipfile
import io
import random
import numpy as np
import librosa
import soundata
import torch
import torchaudio.transforms as T
from scipy.signal import fftconvolve

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
DURATION_SEC   = 3.0
TARGET_LENGTH  = int(SAMPLE_RATE * DURATION_SEC)
N_MELS         = 64
RAW_DIR        = "raw_data"
OUTPUT_DIR     = "prepared_data"
STATS_PATH     = "feature_stats.json"   # saved alongside the model

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
for split in ['train', 'val']:
    for class_id in ['0_safe', '1_unsafe']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_id), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'ravdess'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'tess'), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOADERS
# ─────────────────────────────────────────────────────────────────────────────
def download_ravdess():
    target_dir = os.path.join(RAW_DIR, 'ravdess')
    if len(os.listdir(target_dir)) > 50:
        print("RAVDESS already downloaded. Skipping.")
        return target_dir
    print("Downloading RAVDESS (~200MB)…")
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    r = requests.get(url)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(target_dir)
    print("RAVDESS download complete.")
    return target_dir


def download_tess():
    """
    Toronto Emotional Speech Set — 2800 files, 7 emotions, 2 female speakers.
    Much cleaner recording quality than RAVDESS for 'angry' and 'fear' classes.
    https://tspace.library.utoronto.ca/handle/1807/24487
    Mirrored on Kaggle; we pull the Zenodo copy.
    """
    target_dir = os.path.join(RAW_DIR, 'tess')
    if len(os.listdir(target_dir)) > 10:
        print("TESS already downloaded. Skipping.")
        return target_dir
    print("Downloading TESS (~120MB)…")
    url = "https://zenodo.org/record/6944805/files/TESS.zip?download=1"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(target_dir)
        print("TESS download complete.")
    except Exception as e:
        print(f"TESS download failed ({e}). Continuing without it.")
    return target_dir


def initialize_urbansound():
    print("Checking UrbanSound8K (background clutter)…")
    urbansound = soundata.initialize('urbansound8k')
    urbansound.download()
    return urbansound


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def pad_or_truncate(y):
    if len(y) > TARGET_LENGTH:
        start = random.randint(0, len(y) - TARGET_LENGTH)
        return y[start : start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def mix_audio(signal, noise, snr_db=5):
    """Mix foreground signal with background noise at given SNR."""
    signal = pad_or_truncate(signal)
    noise  = pad_or_truncate(noise)
    p_sig  = np.mean(signal ** 2)
    p_noi  = np.mean(noise  ** 2)
    if p_noi == 0:
        return signal
    scale = np.sqrt(p_sig / (10 ** (snr_db / 10)) / p_noi)
    mixed = signal + noise * scale
    peak = np.max(np.abs(mixed))
    return mixed / peak if peak > 0 else mixed


def add_reverb(y, decay=0.4, n_taps=800):
    """
    Simulate room acoustics with a simple decaying exponential IR.
    Adds realism for echo-heavy metro environments without heavy deps.
    """
    ir = np.exp(-np.linspace(0, decay * n_taps, n_taps)) * np.random.randn(n_taps)
    ir /= np.abs(ir).max() + 1e-8
    reverbed = fftconvolve(y, ir, mode='full')[:len(y)]
    peak = np.max(np.abs(reverbed))
    return reverbed / peak if peak > 0 else reverbed


def extract_features(y, augment=True):
    """
    Convert raw audio to a log-Mel spectrogram.

    CRITICAL FIX vs original:
      - Use ref=1.0 (fixed reference) instead of ref=np.max.
        ref=np.max normalises every clip to its own peak, destroying the
        absolute energy information the model needs to tell 'loud screaming'
        from 'quiet conversation'. With ref=1.0, louder signals produce
        higher dB values, which is exactly what we want.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=1024, hop_length=256          # explicit for consistency with inference
    )
    # ── THE FIX ── use a fixed reference so loudness information is preserved
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)

    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]

    # SpecAugment — only during training, not at inference
    if augment:
        if random.random() < 0.5:
            tensor = T.FrequencyMasking(freq_mask_param=10)(tensor)
        if random.random() < 0.5:
            tensor = T.TimeMasking(time_mask_param=20)(tensor)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE STATS — compute mean/std over all TRAINING tensors so inference
# can normalise identically to what the model saw during training.
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_save_stats():
    """
    Walk the training split, compute global mean and std, save to JSON.
    Must be called after all .pt files are written.
    """
    print("\nComputing feature normalisation statistics from training data…")
    all_values = []
    train_root = os.path.join(OUTPUT_DIR, 'train')

    for class_dir in ['0_safe', '1_unsafe']:
        folder = os.path.join(train_root, class_dir)
        for fname in os.listdir(folder):
            if fname.endswith('.pt'):
                t = torch.load(os.path.join(folder, fname), weights_only=True)
                all_values.append(t.numpy().flatten())

    if not all_values:
        print("  Warning: no .pt files found, skipping stats computation.")
        return

    all_values = np.concatenate(all_values)
    mean = float(np.mean(all_values))
    std  = float(np.std(all_values))
    stats = {"mean": mean, "std": std}

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved → {STATS_PATH}  (mean={mean:.2f}, std={std:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# TESS LABEL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def tess_label(filepath):
    """
    TESS filenames end with the emotion word, e.g. 'OAF_bear_angry.wav'.
    Returns True if Unsafe, False if Safe, None to skip.
    """
    name = os.path.splitext(os.path.basename(filepath))[0].lower()
    if name.endswith('angry') or name.endswith('fear'):
        return True   # Unsafe
    if name.endswith('neutral') or name.endswith('happy') or name.endswith('pleasant_surprise'):
        return False  # Safe
    return None       # disgust / sad — skip for sharp binary boundary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Acquire datasets ──────────────────────────────────────────────────
    print("=" * 50)
    print(" PHASE 1 — Data Acquisition")
    print("=" * 50)
    ravdess_dir = download_ravdess()
    tess_dir    = download_tess()
    urbansound  = initialize_urbansound()

    # ── 2. Extract background noise clips ───────────────────────────────────
    print("\n" + "=" * 50)
    print(" PHASE 2 — Background Clutter Extraction")
    print("=" * 50)
    background_audios = []
    safe_bg_labels = ['street_music', 'engine_idling', 'children_playing']

    for clip_id in urbansound.clip_ids:
        clip = urbansound.clip(clip_id)
        try:
            if clip.tags.labels[0] not in safe_bg_labels:
                continue
            y, sr = clip.audio
            if y is None:
                continue
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            background_audios.append(y)
            # Also save clean background as Safe training examples
            split    = 'train' if random.random() < 0.8 else 'val'
            features = extract_features(pad_or_truncate(y), augment=(split == 'train'))
            torch.save(features, os.path.join(OUTPUT_DIR, split, '0_safe', f"bg_{clip_id}.pt"))
        except Exception:
            continue

    if not background_audios:
        print("Warning: no background audio extracted. Mixing skipped.")

    # ── 3. RAVDESS vocal data ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" PHASE 3 — RAVDESS Emotional Speech")
    print("=" * 50)
    counters = {'safe': 0, 'unsafe': 0}

    for root, _, files in os.walk(ravdess_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            parts = file.replace('.wav', '').split('-')
            if len(parts) < 3:
                continue
            emotion = parts[2]
            # 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful
            if emotion in ('05', '06'):
                is_unsafe = True
            elif emotion in ('01', '02', '03'):
                is_unsafe = False
            else:
                continue  # skip sad/disgust

            filepath = os.path.join(root, file)
            y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

            # Optionally add room acoustics for metro realism
            if random.random() < 0.4:
                y = add_reverb(y)

            # Mix with transit noise
            if background_audios:
                noise = random.choice(background_audios)
                y = mix_audio(y, noise, snr_db=random.uniform(0.0, 15.0))
            else:
                y = pad_or_truncate(y)

            split     = 'train' if random.random() < 0.8 else 'val'
            features  = extract_features(y, augment=(split == 'train'))
            class_dir = '1_unsafe' if is_unsafe else '0_safe'
            label_key = 'unsafe' if is_unsafe else 'safe'
            counters[label_key] += 1
            torch.save(features, os.path.join(OUTPUT_DIR, split, class_dir, f"rav_{file}.pt"))

    print(f"  RAVDESS → safe: {counters['safe']}, unsafe: {counters['unsafe']}")

    # ── 4. TESS vocal data ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" PHASE 4 — TESS Emotional Speech")
    print("=" * 50)
    tess_counters = {'safe': 0, 'unsafe': 0}

    for root, _, files in os.walk(tess_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            is_unsafe = tess_label(os.path.join(root, file))
            if is_unsafe is None:
                continue

            y, sr = librosa.load(os.path.join(root, file), sr=SAMPLE_RATE)
            if random.random() < 0.4:
                y = add_reverb(y)
            if background_audios:
                noise = random.choice(background_audios)
                y = mix_audio(y, noise, snr_db=random.uniform(0.0, 15.0))
            else:
                y = pad_or_truncate(y)

            split     = 'train' if random.random() < 0.8 else 'val'
            features  = extract_features(y, augment=(split == 'train'))
            class_dir = '1_unsafe' if is_unsafe else '0_safe'
            label_key = 'unsafe' if is_unsafe else 'safe'
            tess_counters[label_key] += 1
            torch.save(features, os.path.join(OUTPUT_DIR, split, class_dir, f"tess_{file}.pt"))

    print(f"  TESS   → safe: {tess_counters['safe']}, unsafe: {tess_counters['unsafe']}")

    # ── 5. Compute and save normalisation stats ──────────────────────────────
    compute_and_save_stats()

    print("\n" + "=" * 50)
    print(" DONE — Data ready for training.")
    print(f" Check '{OUTPUT_DIR}/' for .pt tensors and '{STATS_PATH}' for normalisation stats.")
    print("=" * 50)


if __name__ == "__main__":
    main()