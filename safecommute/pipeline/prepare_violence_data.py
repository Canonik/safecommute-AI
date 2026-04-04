"""
Process the HuggingFace violence detection dataset into training .pt tensors.
Label 1 (violent) → unsafe, Label 0 (non-violent) → safe.

Split is determined per SOURCE FILE (not per chunk) using sha256 hash
to prevent data leakage. All chunks from the same file go to the same split.
All spectrograms are saved clean (no augmentation).
"""
import os
import sys

import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, DATA_DIR
from safecommute.features import extract_features, pad_or_truncate, chunk_audio
from safecommute.utils import seed_everything, sha256_split


def main():
    seed_everything()
    violence_dir = os.path.join('raw_data', 'violence')
    if not os.path.exists(violence_dir):
        print("No violence data found")
        return

    total_safe = 0
    total_unsafe = 0

    print("=" * 50)
    print(" Processing Violence Dataset")
    print(" (source-level sha256 split, no augmentation)")
    print("=" * 50)

    for fname in sorted(os.listdir(violence_dir)):
        if not fname.endswith('.wav'):
            continue
        # Parse label from filename: violence_IDX_LABEL.wav
        parts = fname.replace('.wav', '').split('_')
        if len(parts) < 3:
            continue
        label = int(parts[2])  # 0=non-violent, 1=violent

        path = os.path.join(violence_dir, fname)
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            if len(y) < SAMPLE_RATE:
                continue

            # Split by SOURCE FILE — all chunks go to the same split
            split = sha256_split(fname)
            cls = '1_unsafe' if label == 1 else '0_safe'
            out_dir = os.path.join(DATA_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            chunks = chunk_audio(y)
            for i, chunk in enumerate(chunks):
                features = extract_features(chunk)
                base = fname.replace('.wav', '')
                out_path = os.path.join(out_dir, f"viol_{base}_c{i:03d}.pt")
                torch.save(features, out_path)

                if label == 1:
                    total_unsafe += 1
                else:
                    total_safe += 1
        except Exception:
            continue

    print(f"  Violence dataset: {total_safe} safe + {total_unsafe} unsafe chunks")

    # Updated counts
    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            count = len([f for f in os.listdir(folder) if f.endswith('.pt')])
            viol = len([f for f in os.listdir(folder) if f.startswith('viol_')])
            print(f"  {split}/{cls}: {count} total ({viol} from violence dataset)")


if __name__ == "__main__":
    main()
