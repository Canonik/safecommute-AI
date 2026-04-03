"""
Process the HuggingFace violence detection dataset into training .pt tensors.
Label 1 (violent) → unsafe, Label 0 (non-violent) → safe.
"""
import os
import sys
import random
import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, TARGET_LENGTH, DATA_DIR
from safecommute.features import extract_features, pad_or_truncate
from safecommute.utils import seed_everything


def chunk_audio(y, sr=SAMPLE_RATE, chunk_sec=3.0, hop_sec=1.5):
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    chunks = []
    for start in range(0, len(y) - chunk_len + 1, hop_len):
        chunks.append(y[start:start + chunk_len])
    if not chunks and len(y) > sr:
        chunks.append(pad_or_truncate(y))
    return chunks


def main():
    seed_everything()
    violence_dir = os.path.join('raw_data', 'violence')
    if not os.path.exists(violence_dir):
        print("No violence data found")
        return

    total_safe = 0
    total_unsafe = 0

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
            chunks = chunk_audio(y)
            for i, chunk in enumerate(chunks):
                r = random.random()
                split = 'train' if r < 0.70 else ('val' if r < 0.85 else 'test')
                cls = '1_unsafe' if label == 1 else '0_safe'
                out_dir = os.path.join(DATA_DIR, split, cls)
                os.makedirs(out_dir, exist_ok=True)

                features = extract_features(chunk, augment=(split == 'train'))
                base = fname.replace('.wav', '')
                out_path = os.path.join(out_dir, f"viol_{base}_c{i:03d}.pt")
                torch.save(features, out_path)

                if label == 1:
                    total_unsafe += 1
                else:
                    total_safe += 1
        except Exception:
            continue

    print(f"Violence dataset processed: {total_safe} safe + {total_unsafe} unsafe chunks")

    # Updated counts
    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            count = len([f for f in os.listdir(folder) if f.endswith('.pt')])
            viol = len([f for f in os.listdir(folder) if f.startswith('viol_')])
            print(f"  {split}/{cls}: {count} total ({viol} from violence dataset)")


if __name__ == "__main__":
    main()
