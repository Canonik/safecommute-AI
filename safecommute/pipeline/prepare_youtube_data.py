"""
Process YouTube audio into training-ready .pt tensors.
Chunks long audio into 3-second windows with overlap.
Metro ambient → safe class, screams → unsafe class.

Split is determined per SOURCE FILE (not per chunk) using sha256 hash
to prevent data leakage. All chunks from the same file go to the same split.
All spectrograms are saved clean (no augmentation).
"""
import os
import sys
import hashlib

import numpy as np
import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, TARGET_LENGTH, DATA_DIR
from safecommute.features import extract_features, pad_or_truncate
from safecommute.utils import seed_everything


def sha256_split(filename):
    """Deterministic split based on sha256 hash. 70/15/15 train/val/test."""
    h = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
    if h < 70:
        return 'train'
    elif h < 85:
        return 'val'
    else:
        return 'test'


def chunk_audio(y, sr=SAMPLE_RATE, chunk_sec=3.0, hop_sec=1.5):
    """Chunk long audio into overlapping windows."""
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    chunks = []
    for start in range(0, len(y) - chunk_len + 1, hop_len):
        chunks.append(y[start:start + chunk_len])
    if not chunks and len(y) > sr:  # at least 1 second
        chunks.append(pad_or_truncate(y))
    return chunks


def process_directory(audio_dir, label, prefix, output_base):
    """Process all wav files in a directory into .pt feature tensors."""
    total = 0
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith('.wav'):
            continue
        path = os.path.join(audio_dir, fname)
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            if len(y) < SAMPLE_RATE:  # skip < 1 second
                continue

            # Split by SOURCE FILE — all chunks go to the same split
            split = sha256_split(fname)
            cls = '1_unsafe' if label == 1 else '0_safe'
            out_dir = os.path.join(output_base, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            chunks = chunk_audio(y)
            for i, chunk in enumerate(chunks):
                features = extract_features(chunk, augment=False)
                base = fname.replace('.wav', '')
                out_path = os.path.join(out_dir, f"{prefix}_{base}_c{i:03d}.pt")
                torch.save(features, out_path)
                total += 1
        except Exception as e:
            print(f"  Skip {fname}: {e}")
    return total


def main():
    seed_everything()

    metro_dir = os.path.join('raw_data', 'youtube_metro')
    scream_dir = os.path.join('raw_data', 'youtube_screams')

    print("=" * 50)
    print(" Processing YouTube Audio")
    print(" (source-level sha256 split, no augmentation)")
    print("=" * 50)

    # Process metro ambient → safe (label=0)
    if os.path.exists(metro_dir):
        n = process_directory(metro_dir, label=0, prefix='yt_metro', output_base=DATA_DIR)
        print(f"  Metro ambient: {n} chunks → safe class")
    else:
        print("  No metro audio found")

    # Process screams → unsafe (label=1)
    if os.path.exists(scream_dir):
        n = process_directory(scream_dir, label=1, prefix='yt_scream', output_base=DATA_DIR)
        print(f"  Screams/shouts: {n} chunks → unsafe class")
    else:
        print("  No scream audio found")

    # Count results
    print("\n  Updated dataset counts:")
    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) if f.endswith('.pt')])
                yt_count = len([f for f in os.listdir(folder) if f.startswith('yt_')])
                print(f"    {split}/{cls}: {count} total ({yt_count} from YouTube)")

    print("\nDone.")


if __name__ == "__main__":
    main()
