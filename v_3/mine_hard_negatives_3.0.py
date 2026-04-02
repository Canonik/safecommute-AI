"""
mine_hard_negatives.py
─────────────────────────────────────────────────────────────────────────────
Hard negative mining: find audio the model confidently misclassifies as
'unsafe' but which is actually safe, then add those clips to the safe
training class so the model learns from its own mistakes.

This is the highest-ROI improvement you can make to reduce false positives
without collecting new data from scratch.

Usage:
    # Mine from a directory of safe audio files you provide:
    python mine_hard_negatives.py --source /path/to/safe_audio/

    # Or mine from your existing prepared_data safe val set (useful baseline):
    python mine_hard_negatives.py --source prepared_data/val/0_safe --from_pt

    # Set a custom confidence threshold (default 0.55 — any clip scored above
    # this as 'unsafe' is considered a hard negative):
    python mine_hard_negatives.py --source /path/to/audio --threshold 0.60

After running, re-run train_model.py. The new hard negatives will be included
automatically because they are saved to prepared_data/train/0_safe/.
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — must match train_model.py and data_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
STATS_PATH  = "feature_stats.json"
MODEL_PATH  = "safecommute_edge_model.pth"
OUTPUT_DIR  = "prepared_data"
SAMPLE_RATE = 16000
N_MELS      = 64
TIME_FRAMES = 188
HOP_LENGTH  = 256
N_FFT       = 1024
TARGET_LEN  = int(SAMPLE_RATE * 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (must stay in sync with train_model.py)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, pool=(2, 2)):
        return F.avg_pool2d(self.net(x), pool)


class SafeCommuteCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.bn_input    = nn.BatchNorm2d(1)
        self.block1      = ConvBlock(1,   64)
        self.block2      = ConvBlock(64,  128)
        self.block3      = ConvBlock(128, 256)
        freq_dim         = n_mels // (2 ** 3)
        self.freq_reduce = nn.Linear(256 * freq_dim, 256)
        self.gru         = nn.GRU(256, 128, batch_first=True)
        self.dropout     = nn.Dropout(0.3)
        self.fc          = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        x = F.relu(self.freq_reduce(x))
        _, h = self.gru(x)
        return self.fc(self.dropout(h.squeeze(0)))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (must match data_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
def extract_tensor(y, mean, std):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    tensor  = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

    t = tensor.shape[-1]
    if t > TIME_FRAMES:
        tensor = tensor[:, :, :TIME_FRAMES]
    elif t < TIME_FRAMES:
        tensor = torch.cat([tensor, torch.zeros(1, N_MELS, TIME_FRAMES - t)], dim=-1)

    return (tensor - mean) / (std + 1e-8)


def pad_or_truncate(y):
    if len(y) > TARGET_LEN:
        return y[:TARGET_LEN]
    return np.pad(y, (0, TARGET_LEN - len(y)), 'constant')


# ─────────────────────────────────────────────────────────────────────────────
# MINING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def mine_from_audio_files(source_dir, model, mean, std, threshold, device):
    """
    Walk source_dir, load .wav/.mp3/.flac files, score each clip,
    return list of (tensor, filename, unsafe_prob) for clips above threshold.
    """
    found    = []
    scanned  = 0
    exts     = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    for root, _, files in os.walk(source_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            path = os.path.join(root, fname)
            try:
                y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                y     = pad_or_truncate(y)
                feat  = extract_tensor(y, mean, std).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(feat), dim=1)[0][1].item()
                scanned += 1
                if prob >= threshold:
                    # High unsafe score on a file you've asserted is safe = hard negative
                    found.append((extract_tensor(y, mean, std), fname, prob))
                if scanned % 50 == 0:
                    print(f"  Scanned {scanned} files, found {len(found)} hard negatives so far…")
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    return found, scanned


def mine_from_pt_files(source_dir, model, mean, std, threshold, device):
    """
    Directly score pre-computed .pt tensors (e.g. from prepared_data/val/0_safe).
    Useful for finding what the current model gets wrong on existing validation data.
    """
    found   = []
    scanned = 0

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
            inp  = feat_norm.unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(inp), dim=1)[0][1].item()
            scanned += 1
            if prob >= threshold:
                found.append((feat_norm, fname, prob))
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    return found, scanned


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(source_dir, threshold, from_pt):
    # ── Load stats ────────────────────────────────────────────────────────
    if not os.path.exists(STATS_PATH):
        print(f"Error: '{STATS_PATH}' not found.")
        sys.exit(1)
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean, std = s['mean'], s['std']

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found.")
        sys.exit(1)
    device = torch.device("cpu")
    model  = SafeCommuteCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded. Scanning '{source_dir}' for false positives…")
    print(f"Threshold: unsafe_prob ≥ {threshold}\n")

    # ── Mine ──────────────────────────────────────────────────────────────
    if from_pt:
        hard_negs, scanned = mine_from_pt_files(source_dir, model, mean, std, threshold, device)
    else:
        hard_negs, scanned = mine_from_audio_files(source_dir, model, mean, std, threshold, device)

    print(f"\nScanned {scanned} files. Found {len(hard_negs)} hard negatives.")

    if not hard_negs:
        print("No hard negatives found at this threshold. "
              "Try lowering --threshold or providing more audio.")
        return

    # Sort by confidence descending so the worst offenders come first
    hard_negs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 hardest negatives (highest unsafe score on safe audio):")
    for tensor, fname, prob in hard_negs[:10]:
        print(f"  {prob:.3f}  {fname}")

    # ── Save to training set ──────────────────────────────────────────────
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
        print("Re-run data_pipeline.py (compute_and_save_stats only) if you want "
              "updated normalisation statistics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard negative mining for SafeCommute AI")
    parser.add_argument('--source', required=True,
                        help='Directory of safe audio files (.wav/.mp3 etc.) or .pt tensors')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Unsafe probability above which a clip is treated as a hard negative (default: 0.55)')
    parser.add_argument('--from_pt', action='store_true',
                        help='Source directory contains .pt tensors instead of raw audio')
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Error: '{args.source}' is not a directory.")
        sys.exit(1)

    main(source_dir=args.source, threshold=args.threshold, from_pt=args.from_pt)