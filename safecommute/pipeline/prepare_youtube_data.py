"""
Process YouTube audio into training-ready .pt tensors.

YouTube data serves two roles in the three-layer data strategy:
  - youtube_screams (Layer 1, unsafe): Real screams/shouts from YouTube
    compilations. These supplement AudioSet threat categories with more
    naturalistic, uncontrolled recordings (vs. AudioSet's 10-second segments).
  - youtube_metro (Layer 2, safe): Metro station ambient recordings used as
    hard negatives. Train announcements, braking sounds, and crowd noise
    that could confuse a naive classifier.

Chunking strategy: long recordings (often 5-30 minutes) are split into
3-second windows with 50% overlap (1.5s hop). The overlap ensures every
threat event is fully captured in at least one chunk. All chunks from
the same source file are assigned to the same train/val/test split via
sha256 hash to prevent temporal leakage (adjacent chunks share ~50% audio).

All spectrograms are saved clean — no augmentation at prep time.
"""
import os
import sys

import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, DATA_DIR
from safecommute.features import extract_features, pad_or_truncate, chunk_audio
from safecommute.utils import seed_everything, sha256_split


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
                features = extract_features(chunk)
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
