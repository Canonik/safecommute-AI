"""
Pre-compute teacher soft labels from PANNs CNN14 for knowledge distillation.

Usage:
    python -m safecommute.distill

Walks all .pt feature files in prepared_data/, finds the corresponding raw
audio, runs it through PANNs CNN14, and saves the teacher's binary soft
logits as {name}_teacher.pt alongside each feature file.

GDPR: Raw audio is loaded into RAM, scored, then discarded. Only
non-reconstructible soft label tensors are persisted.
"""

import os
import json
import glob

import librosa
import numpy as np
import torch

from safecommute.constants import (
    SAMPLE_RATE, TARGET_LENGTH, DATA_DIR, RAW_DIR, STATS_PATH,
)
from safecommute.features import pad_or_truncate

# AudioSet class indices that map to "unsafe" (escalation/distress).
# Based on the AudioSet ontology: https://research.google.com/audioset/ontology/
UNSAFE_AUDIOSET_LABELS = {
    'Screaming', 'Shout', 'Yell', 'Crying, sobbing',
    'Whimper', 'Groan', 'Battle cry',
    'Baby cry, infant cry',
}


def load_panns_teacher(device='cpu'):
    """Load PANNs CNN14 for AudioTagging."""
    try:
        from panns_inference import AudioTagging
    except ImportError:
        print("Error: panns_inference not installed. Run: pip install panns_inference")
        raise

    at = AudioTagging(checkpoint_path=None, device=device)
    print("PANNs CNN14 teacher loaded.")
    return at


def get_unsafe_indices(at):
    """Find AudioSet class indices that correspond to unsafe labels."""
    labels = at.labels  # list of 527 AudioSet class names
    indices = []
    for i, label in enumerate(labels):
        if label in UNSAFE_AUDIOSET_LABELS:
            indices.append(i)
            print(f"  Unsafe class [{i}]: {label}")
    print(f"  Total unsafe classes: {len(indices)}")
    return indices


def find_raw_audio(pt_filename, raw_dir):
    """
    Attempt to find the raw audio file corresponding to a .pt feature tensor.

    Supports v2 naming conventions:
      as_{category}_{videoid}_{start}_{end}_c{N}.pt → audioset/{threat|safe}/{category}/*.wav
      yt_metro_{id}_c{N}.pt                         → youtube_metro/{id}.wav
      yt_scream_{id}_c{N}.pt                        → youtube_screams/{id}.wav
      viol_{name}_c{N}.pt                           → violence/{name}.wav
      esc_{name}.pt                                 → esc50/audio/{name}.wav
    """
    base = pt_filename.replace('.pt', '')

    if base.startswith('as_'):
        # AudioSet: search both threat and safe directories
        for group in ['threat', 'safe']:
            for path in glob.glob(os.path.join(raw_dir, 'audioset', group, '**', '*.wav'), recursive=True):
                wav_base = os.path.basename(path).replace('.wav', '')
                if wav_base in base:
                    return path

    elif base.startswith('yt_metro_'):
        wav_name = base.replace('yt_metro_', '').rsplit('_c', 1)[0] + '.wav'
        path = os.path.join(raw_dir, 'youtube_metro', wav_name)
        if os.path.exists(path):
            return path

    elif base.startswith('yt_scream_'):
        wav_name = base.replace('yt_scream_', '').rsplit('_c', 1)[0] + '.wav'
        path = os.path.join(raw_dir, 'youtube_screams', wav_name)
        if os.path.exists(path):
            return path

    elif base.startswith('viol_'):
        wav_name = base.rsplit('_c', 1)[0].replace('viol_', '') + '.wav'
        path = os.path.join(raw_dir, 'violence', wav_name)
        if os.path.exists(path):
            return path

    elif base.startswith('esc_'):
        wav_name = base.replace('esc_', '').replace('hns_', '') + '.wav'
        path = os.path.join(raw_dir, 'esc50', 'audio', wav_name)
        if os.path.exists(path):
            return path

    return None


def compute_teacher_label(at, waveform, unsafe_indices, device='cpu'):
    """
    Run audio through PANNs and return binary soft logits [safe_logit, unsafe_logit].
    """
    # PANNs expects (batch, samples) at 32kHz
    if len(waveform) < 1000:
        return None

    # Resample to 32kHz (PANNs native rate)
    waveform_32k = librosa.resample(waveform, orig_sr=SAMPLE_RATE, target_sr=32000)
    audio_input = waveform_32k[np.newaxis, :]  # (1, samples)

    clipwise_output, _ = at.inference(audio_input)
    probs = clipwise_output[0]  # (527,) sigmoid probabilities

    # Aggregate unsafe probability: max over unsafe classes
    unsafe_prob = max(probs[i] for i in unsafe_indices)
    safe_prob = 1.0 - unsafe_prob

    # Return as logits (inverse sigmoid) for KL divergence training
    eps = 1e-7
    safe_logit = np.log(safe_prob + eps) - np.log(1 - safe_prob + eps)
    unsafe_logit = np.log(unsafe_prob + eps) - np.log(1 - unsafe_prob + eps)

    return torch.tensor([safe_logit, unsafe_logit], dtype=torch.float32)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    at = load_panns_teacher(device)
    unsafe_indices = get_unsafe_indices(at)

    if not unsafe_indices:
        print("Error: No unsafe AudioSet classes found. Check UNSAFE_AUDIOSET_LABELS.")
        return

    processed = 0
    skipped = 0
    errors = 0

    for split in ['train', 'val', 'test']:
        for cls in ['0_safe', '1_unsafe']:
            folder = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(folder):
                continue

            pt_files = [f for f in os.listdir(folder)
                        if f.endswith('.pt') and not f.endswith('_teacher.pt')]

            print(f"\nProcessing {split}/{cls}: {len(pt_files)} files")

            for fname in pt_files:
                teacher_path = os.path.join(folder, fname.replace('.pt', '_teacher.pt'))
                if os.path.exists(teacher_path):
                    skipped += 1
                    continue

                raw_path = find_raw_audio(fname, RAW_DIR)
                if raw_path is None:
                    skipped += 1
                    continue

                try:
                    y, _ = librosa.load(raw_path, sr=SAMPLE_RATE, mono=True)
                    y = pad_or_truncate(y)
                    label = compute_teacher_label(at, y, unsafe_indices, device)
                    if label is not None:
                        torch.save(label, teacher_path)
                        processed += 1
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1

                if (processed + skipped + errors) % 200 == 0:
                    print(f"  Progress: {processed} done, {skipped} skipped, {errors} errors")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
