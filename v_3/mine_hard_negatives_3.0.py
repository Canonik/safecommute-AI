"""
mine_hard_negatives.py — find safe audio the model misclassifies as unsafe.

Usage:
    python mine_hard_negatives_3.0.py --source /path/to/safe_audio/
    python mine_hard_negatives_3.0.py --source prepared_data/val/0_safe --from_pt
    python mine_hard_negatives_3.0.py --source /path/to/audio --threshold 0.60
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.features import pad_or_truncate, extract_features
from safecommute.constants import (
    SAMPLE_RATE, N_MELS, TIME_FRAMES, N_FFT, HOP_LENGTH,
    STATS_PATH, MODEL_SAVE_PATH as MODEL_PATH, DATA_DIR as OUTPUT_DIR,
)

TARGET_LEN = int(SAMPLE_RATE * 3.0)


def extract_tensor(y, mean, std):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

    t = tensor.shape[-1]
    if t > TIME_FRAMES:
        tensor = tensor[:, :, :TIME_FRAMES]
    elif t < TIME_FRAMES:
        tensor = torch.cat([tensor, torch.zeros(1, N_MELS, TIME_FRAMES - t)], dim=-1)

    return (tensor - mean) / (std + 1e-8)


def mine_from_audio_files(source_dir, model, mean, std, threshold, device):
    found, scanned = [], 0
    exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            path = os.path.join(root, fname)
            try:
                y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                y = pad_or_truncate(y)
                feat = extract_tensor(y, mean, std).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(feat), dim=1)[0][1].item()
                scanned += 1
                if prob >= threshold:
                    found.append((extract_tensor(y, mean, std), fname, prob))
                if scanned % 50 == 0:
                    print(f"  Scanned {scanned} files, found {len(found)} hard negatives so far…")
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
    return found, scanned


def mine_from_pt_files(source_dir, model, mean, std, threshold, device):
    found, scanned = [], 0
    for fname in os.listdir(source_dir):
        if not fname.endswith('.pt'):
            continue
        path = os.path.join(source_dir, fname)
        try:
            feat = torch.load(path, weights_only=True)
            t = feat.shape[-1]
            if t > TIME_FRAMES:
                feat = feat[:, :, :TIME_FRAMES]
            elif t < TIME_FRAMES:
                feat = torch.cat([feat, torch.zeros(1, N_MELS, TIME_FRAMES - t)], dim=-1)
            feat_norm = (feat - mean) / (std + 1e-8)
            inp = feat_norm.unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(inp), dim=1)[0][1].item()
            scanned += 1
            if prob >= threshold:
                found.append((feat_norm, fname, prob))
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    return found, scanned


def main(source_dir, threshold, from_pt):
    if not os.path.exists(STATS_PATH):
        print(f"Error: '{STATS_PATH}' not found.")
        sys.exit(1)
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found.")
        sys.exit(1)
    device = torch.device("cpu")
    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded. Scanning '{source_dir}' for false positives…")
    print(f"Threshold: unsafe_prob ≥ {threshold}\n")

    if from_pt:
        hard_negs, scanned = mine_from_pt_files(source_dir, model, mean, std, threshold, device)
    else:
        hard_negs, scanned = mine_from_audio_files(source_dir, model, mean, std, threshold, device)

    print(f"\nScanned {scanned} files. Found {len(hard_negs)} hard negatives.")

    if not hard_negs:
        print("No hard negatives found. Try lowering --threshold or providing more audio.")
        return

    hard_negs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 hardest negatives:")
    for tensor, fname, prob in hard_negs[:10]:
        print(f"  {prob:.3f}  {fname}")

    save_dir = os.path.join(OUTPUT_DIR, 'train', '0_safe')
    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    for tensor, fname, prob in hard_negs:
        save_name = f"mined_{fname.replace(os.sep, '_')}.pt"
        save_path = os.path.join(save_dir, save_name)
        if not os.path.exists(save_path):
            torch.save(tensor, save_path)
            saved += 1

    print(f"\nSaved {saved} new hard negatives to '{save_dir}'.")
    if saved > 0:
        print("Re-run train_model.py to incorporate them into training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard negative mining for SafeCommute AI")
    parser.add_argument('--source', required=True,
                        help='Directory of safe audio files or .pt tensors')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Unsafe probability threshold (default: 0.55)')
    parser.add_argument('--from_pt', action='store_true',
                        help='Source contains .pt tensors instead of raw audio')
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Error: '{args.source}' is not a directory.")
        sys.exit(1)

    main(source_dir=args.source, threshold=args.threshold, from_pt=args.from_pt)
