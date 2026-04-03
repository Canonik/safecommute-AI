import os
import sys
import json
import requests
import zipfile
import io
import random
import numpy as np
import librosa
import soundata
import torch
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import (
    SAMPLE_RATE, DURATION_SEC, TARGET_LENGTH, N_MELS, N_FFT, HOP_LENGTH,
    RAW_DIR, DATA_DIR as OUTPUT_DIR, STATS_PATH, SEED,
)
from safecommute.features import extract_features, pad_or_truncate
from safecommute.utils import seed_everything

try:
    import pyroomacoustics as pra
    HAS_PRA = True
    print("pyroomacoustics found — using RIR-based reverberation.")
except ImportError:
    HAS_PRA = False
    print("pyroomacoustics not found — using fallback reverb.")

# ─────────────────────────────────────────────────────────────────────────────
# LABEL CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
# UrbanSound8K
SAFE_BG_LABELS = ['street_music', 'engine_idling', 'children_playing']
HARD_NEG_LABELS = ['jackhammer', 'drilling', 'air_conditioner', 'car_horn']

# ESC-50
ESC_SAFE_AMBIENT = [
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
    'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'clock_alarm',
    'train', 'helicopter', 'church_bells', 'airplane', 'washing_machine',
]
ESC_HARD_NEG = ['siren', 'fireworks', 'chainsaw', 'thunderstorm', 'hand_saw']

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT STRUCTURE (70/15/15 train/val/test)
# ─────────────────────────────────────────────────────────────────────────────
for split in ['train', 'val', 'test']:
    for class_id in ['0_safe', '1_unsafe']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_id), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'ravdess'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'tess'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'cremad'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'savee'), exist_ok=True)


def random_split():
    """70/15/15 train/val/test split."""
    r = random.random()
    if r < 0.70:
        return 'train'
    elif r < 0.85:
        return 'val'
    else:
        return 'test'


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADERS
# ─────────────────────────────────────────────────────────────────────────────
def download_ravdess():
    target_dir = os.path.join(RAW_DIR, 'ravdess')
    if len(os.listdir(target_dir)) > 50:
        print("RAVDESS already downloaded.")
        return target_dir
    print("Downloading RAVDESS (~200MB)…")
    r = requests.get(
        "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        timeout=300)
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
        r = requests.get(
            "https://tspace.library.utoronto.ca/bitstream/1807/24487/2/TESS.zip",
            timeout=300)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(target_dir)
        print("TESS done.")
    except Exception as e:
        print(f"TESS download failed ({e}). Continuing without it.")
    return target_dir


def download_cremad():
    """Download CREMA-D: 7442 clips from 91 actors, 6 emotions via HuggingFace."""
    target_dir = os.path.join(RAW_DIR, 'cremad')
    # Check for real wav files (not LFS pointers)
    real_wavs = []
    for f in os.listdir(target_dir):
        if f.endswith('.wav'):
            fpath = os.path.join(target_dir, f)
            if os.path.getsize(fpath) > 1000:  # LFS pointers are ~130 bytes
                real_wavs.append(f)
    if len(real_wavs) > 100:
        print(f"CREMA-D already downloaded ({len(real_wavs)} files).")
        return target_dir

    # Clean up LFS pointers if present
    for f in os.listdir(target_dir):
        fpath = os.path.join(target_dir, f)
        if f.endswith('.wav') and os.path.getsize(fpath) < 1000:
            os.remove(fpath)

    print("Downloading CREMA-D via HuggingFace datasets…")
    try:
        from datasets import load_dataset
        import soundfile as sf
        ds = load_dataset("TemporalReasoning/CREMA-D", split="train", trust_remote_code=True)
        saved = 0
        for item in ds:
            audio = item.get('audio')
            fname = item.get('file', item.get('path', f'cremad_{saved}'))
            if audio is None:
                continue
            wav_name = os.path.basename(str(fname))
            if not wav_name.endswith('.wav'):
                wav_name = f"cremad_{saved}.wav"
            out_path = os.path.join(target_dir, wav_name)
            if not os.path.exists(out_path):
                sr = audio.get('sampling_rate', 16000)
                arr = np.array(audio['array'], dtype=np.float32)
                sf.write(out_path, arr, sr)
                saved += 1
                if saved % 500 == 0:
                    print(f"  Saved {saved} clips…")
        print(f"CREMA-D done ({saved} clips saved).")
    except Exception as e:
        print(f"CREMA-D download failed ({e}). Continuing without it.")
    return target_dir


def initialize_urbansound():
    print("Checking UrbanSound8K…")
    us = soundata.initialize('urbansound8k')
    us.download()
    return us


def initialize_esc50():
    """Initialize ESC-50 dataset via soundata."""
    print("Checking ESC-50…")
    try:
        esc = soundata.initialize('esc50')
        esc.download()
        return esc
    except Exception as e:
        print(f"ESC-50 download failed ({e}). Continuing without it.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def mix_audio(signal, noise, snr_db=5):
    signal = pad_or_truncate(signal)
    noise = pad_or_truncate(noise)
    p_sig = np.mean(signal ** 2)
    p_noi = np.mean(noise ** 2)
    if p_noi == 0:
        return signal
    scale = np.sqrt(p_sig / (10 ** (snr_db / 10)) / p_noi)
    mixed = signal + noise * scale
    peak = np.max(np.abs(mixed))
    return mixed / peak if peak > 0 else mixed


def add_reverb_simple(y, decay=0.4, n_taps=800):
    ir = np.exp(-np.linspace(0, decay * n_taps, n_taps)) * np.random.randn(n_taps)
    ir /= np.abs(ir).max() + 1e-8
    out = fftconvolve(y, ir, mode='full')[:len(y)]
    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def add_rir_reverb(y):
    if not HAS_PRA:
        return add_reverb_simple(y)
    try:
        rx = random.uniform(5.0, 22.0)
        ry = random.uniform(2.5, 4.5)
        rz = random.uniform(2.2, 3.5)
        absorption = random.uniform(0.05, 0.25)
        room = pra.ShoeBox(
            [rx, ry, rz], fs=SAMPLE_RATE,
            materials=pra.Material(absorption),
            max_order=12, ray_tracing=False, air_absorption=True)
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
# LABEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def tess_label(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0].lower()
    if name.endswith('angry') or name.endswith('fear'):
        return True
    if name.endswith('neutral') or name.endswith('happy') or name.endswith('pleasant_surprise'):
        return False
    return None


def cremad_label(filename):
    """
    CREMA-D filename: {ActorID}_{Sentence}_{Emotion}_{EmotionLevel}.wav
    Emotions: ANG, DIS, FEA, HAP, NEU, SAD
    """
    parts = filename.replace('.wav', '').split('_')
    if len(parts) < 3:
        return None
    emotion = parts[2]
    if emotion in ('ANG', 'FEA'):
        return True   # unsafe
    elif emotion in ('HAP', 'NEU', 'SAD'):
        return False  # safe
    return None  # DIS (disgust) excluded — ambiguous


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    seed_everything()

    print("=" * 50)
    print(" PHASE 1 — Data Acquisition")
    print("=" * 50)
    ravdess_dir = download_ravdess()
    tess_dir = download_tess()
    cremad_dir = download_cremad()
    urbansound = initialize_urbansound()
    esc50 = initialize_esc50()

    print("\n" + "=" * 50)
    print(" PHASE 2 — Background Clutter + Hard Negatives (UrbanSound8K)")
    print("=" * 50)
    background_audios = []
    hard_neg_count = 0

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
            split = random_split()
            features = extract_features(y, augment=(split == 'train'))

            if label in SAFE_BG_LABELS:
                background_audios.append(y)
                torch.save(features, os.path.join(
                    OUTPUT_DIR, split, '0_safe', f"bg_{clip_id}.pt"))
            elif label in HARD_NEG_LABELS:
                torch.save(features, os.path.join(
                    OUTPUT_DIR, split, '0_safe', f"hns_{clip_id}.pt"))
                hard_neg_count += 1
        except Exception:
            continue

    print(f"  Background clips:  {len(background_audios)}")
    print(f"  Hard negatives:    {hard_neg_count}")
    if not background_audios:
        print("  Warning: no background audio. Mixing skipped.")

    # ── ESC-50 ────────────────────────────────────────────────────────────
    esc_count = 0
    if esc50 is not None:
        print("\n" + "=" * 50)
        print(" PHASE 2b — ESC-50 Environmental Sounds")
        print("=" * 50)
        for clip_id in esc50.clip_ids:
            try:
                clip = esc50.clip(clip_id)
                category = clip.category
                if category not in ESC_SAFE_AMBIENT + ESC_HARD_NEG:
                    continue
                y, sr = clip.audio
                if y is None:
                    continue
                if sr != SAMPLE_RATE:
                    y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
                if len(y.shape) > 1:
                    y = librosa.to_mono(y)
                y = pad_or_truncate(y)
                split = random_split()
                features = extract_features(y, augment=(split == 'train'))

                if category in ESC_SAFE_AMBIENT:
                    background_audios.append(y)
                    torch.save(features, os.path.join(
                        OUTPUT_DIR, split, '0_safe', f"esc_{clip_id}.pt"))
                elif category in ESC_HARD_NEG:
                    torch.save(features, os.path.join(
                        OUTPUT_DIR, split, '0_safe', f"esc_hns_{clip_id}.pt"))
                esc_count += 1
            except Exception:
                continue
        print(f"  ESC-50 clips processed: {esc_count}")

    def process_clip(y, is_unsafe, prefix, filename, split=None):
        if random.random() < 0.5:
            y = add_rir_reverb(y)
        if background_audios:
            noise = random.choice(background_audios)
            y = mix_audio(y, noise, snr_db=random.uniform(0.0, 15.0))
        else:
            y = pad_or_truncate(y)
        if split is None:
            split = random_split()
        features = extract_features(y, augment=(split == 'train'))
        class_dir = '1_unsafe' if is_unsafe else '0_safe'
        torch.save(features, os.path.join(
            OUTPUT_DIR, split, class_dir, f"{prefix}_{filename}.pt"))
        return split

    # ── RAVDESS ───────────────────────────────────────────────────────────
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

    # ── TESS ──────────────────────────────────────────────────────────────
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

    # ── CREMA-D ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" PHASE 5 — CREMA-D")
    print("=" * 50)
    cremad_c = {'safe': 0, 'unsafe': 0}
    for root, _, files in os.walk(cremad_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            is_unsafe = cremad_label(file)
            if is_unsafe is None:
                continue
            try:
                y, _ = librosa.load(os.path.join(root, file), sr=SAMPLE_RATE)
                process_clip(y, is_unsafe, 'cremad', file)
                cremad_c['unsafe' if is_unsafe else 'safe'] += 1
            except Exception as e:
                print(f"  Skipping {file}: {e}")
                continue
    print(f"  safe: {cremad_c['safe']}, unsafe: {cremad_c['unsafe']}")

    # ── SAVEE ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" PHASE 6 — SAVEE")
    print("=" * 50)
    savee_dir = os.path.join(RAW_DIR, 'savee')
    savee_c = {'safe': 0, 'unsafe': 0}
    # SAVEE emotion labels are in filename: DC_a01.wav (a=anger, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise)
    # Also handles HuggingFace format: savee_N_LABEL.wav
    for root, _, files in os.walk(savee_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            name_lower = file.lower()
            is_unsafe = None
            # HuggingFace format: savee_N_angry.wav etc.
            if 'angry' in name_lower or 'anger' in name_lower or 'fear' in name_lower:
                is_unsafe = True
            elif 'happy' in name_lower or 'neutral' in name_lower or 'sad' in name_lower or 'calm' in name_lower:
                is_unsafe = False
            # Original SAVEE format: XX_aNNN.wav (a=anger), XX_fNNN.wav (f=fear)
            elif '_a' in name_lower and not '_an' in name_lower:
                is_unsafe = True
            elif '_f' in name_lower:
                is_unsafe = True
            elif '_h' in name_lower or '_n' in name_lower or '_sa' in name_lower:
                is_unsafe = False
            if is_unsafe is None:
                continue
            try:
                y, _ = librosa.load(os.path.join(root, file), sr=SAMPLE_RATE)
                process_clip(y, is_unsafe, 'savee', file)
                savee_c['unsafe' if is_unsafe else 'safe'] += 1
            except Exception as e:
                continue
    print(f"  safe: {savee_c['safe']}, unsafe: {savee_c['unsafe']}")

    # ── ESC-50 (from raw_data/esc50/) ─────────────────────────────────────
    print("\n" + "=" * 50)
    print(" PHASE 7 — ESC-50 (local)")
    print("=" * 50)
    esc_local_dir = os.path.join(RAW_DIR, 'esc50')
    esc_audio_dir = os.path.join(esc_local_dir, 'audio')
    esc_csv_path = os.path.join(esc_local_dir, 'esc50.csv')
    esc_local_count = 0
    if os.path.exists(esc_csv_path) and os.path.exists(esc_audio_dir):
        import pandas as pd
        esc_meta = pd.read_csv(esc_csv_path)
        for _, row in esc_meta.iterrows():
            category = row['category']
            if category not in ESC_SAFE_AMBIENT + ESC_HARD_NEG:
                continue
            wav_path = os.path.join(esc_audio_dir, row['filename'])
            if not os.path.exists(wav_path):
                continue
            try:
                y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
                y = pad_or_truncate(y)
                split = random_split()
                features = extract_features(y, augment=(split == 'train'))
                if category in ESC_SAFE_AMBIENT:
                    background_audios.append(y)
                    torch.save(features, os.path.join(
                        OUTPUT_DIR, split, '0_safe', f"esc_{row['filename'].replace('.wav','')}.pt"))
                elif category in ESC_HARD_NEG:
                    torch.save(features, os.path.join(
                        OUTPUT_DIR, split, '0_safe', f"esc_hns_{row['filename'].replace('.wav','')}.pt"))
                esc_local_count += 1
            except Exception:
                continue
        print(f"  ESC-50 clips processed: {esc_local_count}")
    else:
        print("  ESC-50 not found locally. Run v_3/download_datasets.py first.")

    compute_and_save_stats()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" SUMMARY")
    print("=" * 50)
    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(OUTPUT_DIR, split, cls)
            count = len([f for f in os.listdir(folder) if f.endswith('.pt')])
            print(f"  {split:6}/{cls:10}: {count:>5} samples")

    print("\n" + "=" * 50)
    print(" DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
