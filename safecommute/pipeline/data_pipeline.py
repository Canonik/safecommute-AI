"""
Data preparation pipeline for SafeCommute AI (v2).

Processes raw audio into clean (un-augmented) mel spectrogram .pt tensors,
split into train/val/test with NO data leakage.

Sources:
  - UrbanSound8K (safe): split by predefined folds (1-7/8/9-10)
  - ESC-50 (safe): split by predefined folds (1-3/4/5)
  - AudioSet threat + safe: split by sha256 hash of filename
  - FSD50K threat + safe (fallback): split by sha256 hash of filename

Critical invariants:
  - extract_features() is ALWAYS called with augment=False
  - No mix_audio, no reverb, no SpecAugment during data preparation
  - All augmentation happens at training time (see train.py / dataset.py)
  - Splitting is deterministic and source-aware (no random per-sample splits)
"""

import os
import sys
import json
import hashlib

import numpy as np
import librosa
import pandas as pd
import soundata
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import (
    SAMPLE_RATE, RAW_DIR, DATA_DIR as OUTPUT_DIR, STATS_PATH, SEED,
)
from safecommute.features import extract_features, pad_or_truncate
from safecommute.utils import seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# LABEL CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
# UrbanSound8K categories → safe class
SAFE_BG_LABELS = ['street_music', 'engine_idling', 'children_playing']
HARD_NEG_LABELS = ['jackhammer', 'drilling', 'air_conditioner', 'car_horn']

# ESC-50 categories → safe class
ESC_SAFE_AMBIENT = [
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
    'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'clock_alarm',
    'train', 'helicopter', 'church_bells', 'airplane', 'washing_machine',
]
ESC_HARD_NEG = ['siren', 'fireworks', 'chainsaw', 'thunderstorm', 'hand_saw']


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
for split in ['train', 'val', 'test']:
    for class_id in ['0_safe', '1_unsafe']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_id), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sha256_split(filename):
    """Deterministic split based on sha256 hash. 70/15/15 train/val/test."""
    h = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
    if h < 70:
        return 'train'
    elif h < 85:
        return 'val'
    else:
        return 'test'


def urbansound_fold_to_split(fold):
    """UrbanSound8K: folds 1-7 → train, 8 → val, 9-10 → test."""
    if fold <= 7:
        return 'train'
    elif fold == 8:
        return 'val'
    else:
        return 'test'


def esc50_fold_to_split(fold):
    """ESC-50: folds 1-3 → train, 4 → val, 5 → test."""
    if fold <= 3:
        return 'train'
    elif fold == 4:
        return 'val'
    else:
        return 'test'


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def process_and_save(y, split, class_dir, filename_prefix):
    """Extract clean features and save as .pt tensor."""
    features = extract_features(y, augment=False)
    out_path = os.path.join(OUTPUT_DIR, split, class_dir, f"{filename_prefix}.pt")
    torch.save(features, out_path)
    return split, class_dir


def process_urbansound8k():
    """Process UrbanSound8K using predefined folds for splitting."""
    print("=" * 50)
    print(" UrbanSound8K (safe class, fold-based split)")
    print("=" * 50)

    us = soundata.initialize('urbansound8k')
    counts = {'train': 0, 'val': 0, 'test': 0}

    for clip_id in us.clip_ids:
        try:
            clip = us.clip(clip_id)
            label = clip.tags.labels[0]

            if label not in SAFE_BG_LABELS + HARD_NEG_LABELS:
                continue

            y, sr = clip.audio
            if y is None:
                continue
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            y = pad_or_truncate(y)

            fold = clip.fold
            split = urbansound_fold_to_split(fold)

            prefix = f"hns_{clip_id}" if label in HARD_NEG_LABELS else f"bg_{clip_id}"
            process_and_save(y, split, '0_safe', prefix)
            counts[split] += 1
        except Exception:
            continue

    print(f"  train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts


def process_esc50():
    """Process ESC-50 from local files using predefined folds."""
    print("\n" + "=" * 50)
    print(" ESC-50 (safe class, fold-based split)")
    print("=" * 50)

    esc_dir = os.path.join(RAW_DIR, 'esc50')
    audio_dir = os.path.join(esc_dir, 'audio')
    csv_path = os.path.join(esc_dir, 'esc50.csv')

    if not os.path.exists(csv_path) or not os.path.exists(audio_dir):
        print("  ESC-50 not found. Run download_datasets.py first.")
        return {'train': 0, 'val': 0, 'test': 0}

    meta = pd.read_csv(csv_path)
    counts = {'train': 0, 'val': 0, 'test': 0}
    valid_categories = ESC_SAFE_AMBIENT + ESC_HARD_NEG

    for _, row in meta.iterrows():
        category = row['category']
        if category not in valid_categories:
            continue

        wav_path = os.path.join(audio_dir, row['filename'])
        if not os.path.exists(wav_path):
            continue

        try:
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
            y = pad_or_truncate(y)

            fold = int(row['fold'])
            split = esc50_fold_to_split(fold)

            base = row['filename'].replace('.wav', '')
            prefix = f"esc_hns_{base}" if category in ESC_HARD_NEG else f"esc_{base}"
            process_and_save(y, split, '0_safe', prefix)
            counts[split] += 1
        except Exception:
            continue

    print(f"  train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts


def process_audioset_dir(base_dir, source_name):
    """
    Process AudioSet or FSD50K directory structure:
      base_dir/threat/{category}/*.wav → unsafe (label=1)
      base_dir/safe/{category}/*.wav   → safe   (label=0)

    Split by sha256 hash of filename.
    """
    safe_counts = {'train': 0, 'val': 0, 'test': 0}
    unsafe_counts = {'train': 0, 'val': 0, 'test': 0}

    for group, class_dir, counter in [
        ('threat', '1_unsafe', unsafe_counts),
        ('safe', '0_safe', safe_counts),
    ]:
        group_dir = os.path.join(base_dir, group)
        if not os.path.exists(group_dir):
            continue

        for category in sorted(os.listdir(group_dir)):
            cat_dir = os.path.join(group_dir, category)
            if not os.path.isdir(cat_dir):
                continue

            for fname in os.listdir(cat_dir):
                if not fname.endswith('.wav'):
                    continue

                wav_path = os.path.join(cat_dir, fname)
                try:
                    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
                    if len(y) < SAMPLE_RATE // 2:  # skip < 0.5 seconds
                        continue
                    y = pad_or_truncate(y)

                    split = sha256_split(fname)
                    prefix = f"{source_name}_{category}_{fname.replace('.wav', '')}"
                    process_and_save(y, split, class_dir, prefix)
                    counter[split] += 1
                except Exception:
                    continue

    return safe_counts, unsafe_counts


def process_audioset():
    """Process AudioSet downloaded clips."""
    print("\n" + "=" * 50)
    print(" AudioSet (threat + safe, sha256-based split)")
    print("=" * 50)

    audioset_dir = os.path.join(RAW_DIR, 'audioset')
    if not os.path.exists(audioset_dir):
        print("  AudioSet not found. Run download_audioset.py first.")
        return {'train': 0, 'val': 0, 'test': 0}, {'train': 0, 'val': 0, 'test': 0}

    safe_counts, unsafe_counts = process_audioset_dir(audioset_dir, 'as')
    print(f"  Safe:   train={safe_counts['train']}, val={safe_counts['val']}, test={safe_counts['test']}")
    print(f"  Unsafe: train={unsafe_counts['train']}, val={unsafe_counts['val']}, test={unsafe_counts['test']}")
    return safe_counts, unsafe_counts


def process_fsd50k():
    """Process FSD50K fallback clips (if available)."""
    print("\n" + "=" * 50)
    print(" FSD50K (fallback, sha256-based split)")
    print("=" * 50)

    fsd_dir = os.path.join(RAW_DIR, 'fsd50k')
    if not os.path.exists(fsd_dir):
        print("  FSD50K not found (optional fallback). Skipping.")
        return {'train': 0, 'val': 0, 'test': 0}, {'train': 0, 'val': 0, 'test': 0}

    safe_counts, unsafe_counts = process_audioset_dir(fsd_dir, 'fsd')
    print(f"  Safe:   train={safe_counts['train']}, val={safe_counts['val']}, test={safe_counts['test']}")
    print(f"  Unsafe: train={unsafe_counts['train']}, val={unsafe_counts['val']}, test={unsafe_counts['test']}")
    return safe_counts, unsafe_counts


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE STATS
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_save_stats():
    """Compute global mean/std from training set for normalization."""
    print("\nComputing feature normalization statistics...")
    all_values = []
    for class_dir in ['0_safe', '1_unsafe']:
        folder = os.path.join(OUTPUT_DIR, 'train', class_dir)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith('.pt'):
                t = torch.load(os.path.join(folder, fname), weights_only=True)
                all_values.append(t.numpy().flatten())
    if not all_values:
        print("  Warning: no .pt files found in training set.")
        return
    all_values = np.concatenate(all_values)
    stats = {"mean": float(np.mean(all_values)), "std": float(np.std(all_values))}
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {STATS_PATH}  (mean={stats['mean']:.2f}, std={stats['std']:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    seed_everything()

    print("=" * 60)
    print(" SafeCommute AI — Data Pipeline v2")
    print(" Clean spectrograms, source-aware splits, no augmentation")
    print("=" * 60)

    # Process each source
    us_counts = process_urbansound8k()
    esc_counts = process_esc50()
    as_safe, as_unsafe = process_audioset()
    fsd_safe, fsd_unsafe = process_fsd50k()

    # Compute normalization stats from training set
    compute_and_save_stats()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    # Per-split counts
    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(OUTPUT_DIR, split, cls)
            count = len([f for f in os.listdir(folder) if f.endswith('.pt')])
            print(f"  {split:6}/{cls:10}: {count:>5} samples")

    # Per-source summary table
    print(f"\n  {'Source':<16} {'Safe (tr/va/te)':<22} {'Unsafe (tr/va/te)':<22} {'Split method'}")
    print("  " + "─" * 76)

    def fmt(c):
        return f"{c['train']}/{c['val']}/{c['test']}"

    zero = {'train': 0, 'val': 0, 'test': 0}
    print(f"  {'UrbanSound8K':<16} {fmt(us_counts):<22} {'—':<22} {'predefined folds'}")
    print(f"  {'ESC-50':<16} {fmt(esc_counts):<22} {'—':<22} {'predefined folds'}")
    print(f"  {'AudioSet safe':<16} {fmt(as_safe):<22} {'—':<22} {'sha256 hash'}")
    print(f"  {'AudioSet threat':<16} {'—':<22} {fmt(as_unsafe):<22} {'sha256 hash'}")
    if any(v > 0 for v in fsd_safe.values()) or any(v > 0 for v in fsd_unsafe.values()):
        print(f"  {'FSD50K safe':<16} {fmt(fsd_safe):<22} {'—':<22} {'sha256 hash'}")
        print(f"  {'FSD50K threat':<16} {'—':<22} {fmt(fsd_unsafe):<22} {'sha256 hash'}")

    print("\n  NOTE: YouTube and violence data are processed by separate scripts.")
    print("  Run prepare_youtube_data.py and prepare_violence_data.py next.")

    print("\n" + "=" * 60)
    print(" DONE — All .pt files are clean (un-augmented) spectrograms")
    print("=" * 60)


if __name__ == "__main__":
    main()
