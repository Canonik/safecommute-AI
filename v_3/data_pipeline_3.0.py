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

# Optional: pyroomacoustics gives physically accurate RIR-based reverberation.
# pip install pyroomacoustics
try:
    import pyroomacoustics as pra
    HAS_PRA = True
    print("pyroomacoustics found — using RIR-based reverberation.")
except ImportError:
    HAS_PRA = False
    print("pyroomacoustics not found — using fallback reverb. (pip install pyroomacoustics to improve)")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
DURATION_SEC  = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)
N_MELS        = 64
RAW_DIR       = "raw_data"
OUTPUT_DIR    = "prepared_data"
STATS_PATH    = "feature_stats.json"

# Ambient background — used for SNR mixing
SAFE_BG_LABELS = ['street_music', 'engine_idling', 'children_playing']

# Loud but definitively safe sounds — classic false-positive triggers.
# These are saved to the safe class with prefix hns_ (hard negative sample)
# so the model explicitly learns that high energy != unsafe.
HARD_NEG_LABELS = ['jackhammer', 'drilling', 'air_conditioner', 'car_horn']

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
for split in ['train', 'val']:
    for class_id in ['0_safe', '1_unsafe']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_id), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'ravdess'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'tess'), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADERS
# ─────────────────────────────────────────────────────────────────────────────
def download_ravdess():
    target_dir = os.path.join(RAW_DIR, 'ravdess')
    if len(os.listdir(target_dir)) > 50:
        print("RAVDESS already downloaded.")
        return target_dir
    print("Downloading RAVDESS (~200MB)…")
    r = requests.get("https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1")
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(target_dir)
    print("RAVDESS done.")
    return target_dir


def download_tess():
    target_dir = os.path.join(RAW_DIR, 'tess')
    if len(os.listdir(target_dir)) > 10:
        print("TESS already downloaded.")
        return target_dir
    print("Downloading TESS (~120MB)…")
    try:
        r = requests.get("https://zenodo.org/record/6944805/files/TESS.zip?download=1", timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(target_dir)
        print("TESS done.")
    except Exception as e:
        print(f"TESS download failed ({e}). Continuing without it.")
    return target_dir


def initialize_urbansound():
    print("Checking UrbanSound8K…")
    us = soundata.initialize('urbansound8k')
    us.download()
    return us


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def pad_or_truncate(y):
    if len(y) > TARGET_LENGTH:
        start = random.randint(0, len(y) - TARGET_LENGTH)
        return y[start:start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')


def mix_audio(signal, noise, snr_db=5):
    signal = pad_or_truncate(signal)
    noise  = pad_or_truncate(noise)
    p_sig  = np.mean(signal ** 2)
    p_noi  = np.mean(noise  ** 2)
    if p_noi == 0:
        return signal
    scale = np.sqrt(p_sig / (10 ** (snr_db / 10)) / p_noi)
    mixed = signal + noise * scale
    peak  = np.max(np.abs(mixed))
    return mixed / peak if peak > 0 else mixed


def add_reverb_simple(y, decay=0.4, n_taps=800):
    """Fallback: decaying exponential impulse response."""
    ir = np.exp(-np.linspace(0, decay * n_taps, n_taps)) * np.random.randn(n_taps)
    ir /= np.abs(ir).max() + 1e-8
    out  = fftconvolve(y, ir, mode='full')[:len(y)]
    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def add_rir_reverb(y):
    """
    Simulate metro-like acoustics using a random shoebox room.
    Hard surfaces (tile, metal) → low absorption → long reverb tail.
    Falls back to simple reverb if pyroomacoustics is not installed.
    """
    if not HAS_PRA:
        return add_reverb_simple(y)
    try:
        rx = random.uniform(5.0, 22.0)   # carriage or platform length
        ry = random.uniform(2.5, 4.5)    # width
        rz = random.uniform(2.2, 3.5)    # height
        absorption = random.uniform(0.05, 0.25)   # hard surfaces

        room = pra.ShoeBox(
            [rx, ry, rz],
            fs=SAMPLE_RATE,
            materials=pra.Material(absorption),
            max_order=12,
            ray_tracing=False,
            air_absorption=True,
        )
        src = [random.uniform(0.5, rx - 0.5),
               random.uniform(0.5, ry - 0.5),
               random.uniform(0.8, rz - 0.3)]
        mic = np.array([[random.uniform(0.1, 1.0)],
                        [random.uniform(0.5, ry - 0.5)],
                        [random.uniform(0.5, 1.5)]])
        room.add_source(src, signal=pad_or_truncate(y))
        room.add_microphone(mic)
        room.simulate()
        out = room.mic_array.signals[0]
        out = out[:len(y)] if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out)))
        peak = np.max(np.abs(out))
        return out / peak if peak > 0 else out
    except Exception:
        return add_reverb_simple(y)


def extract_features(y, augment=True):
    """
    Log-Mel spectrogram with ref=1.0 (fixed reference) to preserve loudness.
    SpecAugment applied only when augment=True (i.e. training split).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=1024, hop_length=256
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    tensor  = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

    if augment:
        if random.random() < 0.5:
            tensor = T.FrequencyMasking(freq_mask_param=10)(tensor)
        if random.random() < 0.5:
            tensor = T.TimeMasking(time_mask_param=20)(tensor)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE STATS
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_save_stats():
    print("\nComputing feature normalisation statistics…")
    all_values = []
    for class_dir in ['0_safe', '1_unsafe']:
        folder = os.path.join(OUTPUT_DIR, 'train', class_dir)
        for fname in os.listdir(folder):
            if fname.endswith('.pt'):
                t = torch.load(os.path.join(folder, fname), weights_only=True)
                all_values.append(t.numpy().flatten())
    if not all_values:
        print("  Warning: no .pt files found.")
        return
    all_values = np.concatenate(all_values)
    stats = {"mean": float(np.mean(all_values)), "std": float(np.std(all_values))}
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {STATS_PATH}  (mean={stats['mean']:.2f}, std={stats['std']:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# TESS LABEL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def tess_label(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0].lower()
    if name.endswith('angry') or name.endswith('fear'):
        return True
    if name.endswith('neutral') or name.endswith('happy') or name.endswith('pleasant_surprise'):
        return False
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print(" PHASE 1 — Data Acquisition")
    print("=" * 50)
    ravdess_dir = download_ravdess()
    tess_dir    = download_tess()
    urbansound  = initialize_urbansound()

    print("\n" + "=" * 50)
    print(" PHASE 2 — Background Clutter + Hard Negatives")
    print("=" * 50)
    background_audios = []
    hard_neg_count    = 0

    for clip_id in urbansound.clip_ids:
        clip = urbansound.clip(clip_id)
        try:
            label = clip.tags.labels[0]
            y, sr = clip.audio
            if y is None:
                continue
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            y = pad_or_truncate(y)

            split    = 'train' if random.random() < 0.8 else 'val'
            features = extract_features(y, augment=(split == 'train'))

            if label in SAFE_BG_LABELS:
                background_audios.append(y)
                torch.save(features, os.path.join(OUTPUT_DIR, split, '0_safe', f"bg_{clip_id}.pt"))

            elif label in HARD_NEG_LABELS:
                # Deliberately loud, non-vocal safe sounds to teach the model
                # that raw energy alone is not sufficient for an unsafe prediction.
                torch.save(features, os.path.join(OUTPUT_DIR, split, '0_safe', f"hns_{clip_id}.pt"))
                hard_neg_count += 1

        except Exception:
            continue

    print(f"  Background clips:  {len(background_audios)}")
    print(f"  Hard negatives:    {hard_neg_count}")
    if not background_audios:
        print("  Warning: no background audio. Mixing skipped.")

    def process_clip(y, is_unsafe, prefix, filename, split=None):
        if random.random() < 0.5:
            y = add_rir_reverb(y)
        if background_audios:
            noise = random.choice(background_audios)
            y = mix_audio(y, noise, snr_db=random.uniform(0.0, 15.0))
        else:
            y = pad_or_truncate(y)
        if split is None:
            split = 'train' if random.random() < 0.8 else 'val'
        features  = extract_features(y, augment=(split == 'train'))
        class_dir = '1_unsafe' if is_unsafe else '0_safe'
        torch.save(features, os.path.join(OUTPUT_DIR, split, class_dir, f"{prefix}_{filename}.pt"))
        return split

    print("\n" + "=" * 50)
    print(" PHASE 3 — RAVDESS")
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
            if emotion in ('05', '06'):
                is_unsafe = True
            elif emotion in ('01', '02', '03'):
                is_unsafe = False
            else:
                continue
            y, _ = librosa.load(os.path.join(root, file), sr=SAMPLE_RATE)
            process_clip(y, is_unsafe, 'rav', file)
            counters['unsafe' if is_unsafe else 'safe'] += 1
    print(f"  safe: {counters['safe']}, unsafe: {counters['unsafe']}")

    print("\n" + "=" * 50)
    print(" PHASE 4 — TESS")
    print("=" * 50)
    tess_c = {'safe': 0, 'unsafe': 0}
    for root, _, files in os.walk(tess_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            is_unsafe = tess_label(os.path.join(root, file))
            if is_unsafe is None:
                continue
            y, _ = librosa.load(os.path.join(root, file), sr=SAMPLE_RATE)
            process_clip(y, is_unsafe, 'tess', file)
            tess_c['unsafe' if is_unsafe else 'safe'] += 1
    print(f"  safe: {tess_c['safe']}, unsafe: {tess_c['unsafe']}")

    compute_and_save_stats()
    print("\n" + "=" * 50)
    print(" DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()