"""
Download and prepare high-quality speech data for SafeCommute AI.

Adds diverse speech samples as safe class (label=0) to fix the critical
72% speech false positive rate. Only HIGH QUALITY sources that guarantee
real-world deployment improvement.

Sources:
  1. LibriSpeech clean-100: studio-quality read English speech, 16kHz native
  2. LibriSpeech clean-360: larger set, same quality (optional, for more data)

These are chosen because:
  - 16kHz native sample rate (matches our pipeline exactly)
  - Diverse speakers (251 speakers in clean-100, 921 in clean-360)
  - Clean recording conditions (no music, no noise)
  - Well-validated, widely used in speech research
  - Free, no registration required

Usage:
    PYTHONPATH=. python safecommute/pipeline/download_speech_data.py
    PYTHONPATH=. python safecommute/pipeline/download_speech_data.py --max-samples 20000
    PYTHONPATH=. python safecommute/pipeline/download_speech_data.py --include-360
"""

import os
import sys
import glob
import tarfile
import argparse
import hashlib

import numpy as np
import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, DATA_DIR, RAW_DIR
from safecommute.features import pad_or_truncate, chunk_audio, extract_features
from safecommute.utils import sha256_split


LIBRISPEECH_URL_100 = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_URL_360 = "https://www.openslr.org/resources/12/train-clean-360.tar.gz"

SPEECH_RAW_DIR = os.path.join(RAW_DIR, 'librispeech')


def download_librispeech(url, target_dir):
    """Download and extract a LibriSpeech split."""
    os.makedirs(target_dir, exist_ok=True)

    tarname = os.path.basename(url)
    tarpath = os.path.join(target_dir, tarname)

    # Check if already extracted
    extracted_dir = os.path.join(target_dir, 'LibriSpeech')
    if os.path.exists(extracted_dir):
        flac_count = len(glob.glob(os.path.join(extracted_dir, '**', '*.flac'), recursive=True))
        if flac_count > 100:
            print(f"  Already extracted: {flac_count} FLAC files in {extracted_dir}")
            return extracted_dir

    # Download
    if not os.path.exists(tarpath):
        print(f"  Downloading {tarname} (~6GB for clean-100, ~23GB for clean-360)...")
        import urllib.request
        urllib.request.urlretrieve(url, tarpath, reporthook=_progress)
        print()
    else:
        print(f"  Already downloaded: {tarpath}")

    # Extract
    print(f"  Extracting {tarname}...")
    with tarfile.open(tarpath, 'r:gz') as tar:
        tar.extractall(path=target_dir)

    # Clean up tar
    os.remove(tarpath)
    print(f"  Extracted to {extracted_dir}")
    return extracted_dir


def _progress(count, block_size, total_size):
    """Download progress callback."""
    pct = int(100 * count * block_size / total_size)
    print(f"\r  Progress: {pct}%", end='', flush=True)


def process_speech_files(extracted_dir, max_samples=20000):
    """
    Process LibriSpeech FLAC files into PCEN spectrogram .pt files.

    Chunks long utterances into 3-second windows. Skips silence-heavy chunks.
    Uses sha256-based deterministic split (same as all other sources).
    """
    flac_files = sorted(glob.glob(os.path.join(extracted_dir, '**', '*.flac'), recursive=True))
    print(f"\n  Found {len(flac_files)} FLAC files")

    if not flac_files:
        print("  ERROR: No FLAC files found!")
        return 0

    processed = 0
    skipped_silence = 0
    skipped_short = 0
    errors = 0

    for i, flac_path in enumerate(flac_files):
        if processed >= max_samples:
            print(f"\n  Reached max_samples={max_samples}, stopping.")
            break

        try:
            # Load at 16kHz (LibriSpeech is already 16kHz, so no resampling)
            y, sr = librosa.load(flac_path, sr=SAMPLE_RATE, mono=True)

            if len(y) < SAMPLE_RATE:  # skip < 1 second
                skipped_short += 1
                continue

            # Chunk into 3-second windows
            chunks = chunk_audio(y)

            for ci, chunk in enumerate(chunks):
                if processed >= max_samples:
                    break

                # Skip silence-heavy chunks (RMS < 0.01)
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 0.01:
                    skipped_silence += 1
                    continue

                # Determine split via sha256
                fname_base = os.path.basename(flac_path).replace('.flac', '')
                chunk_id = f"speech_{fname_base}_c{ci:03d}"
                split = sha256_split(chunk_id)

                # Extract PCEN features
                features = extract_features(chunk)

                # Save to appropriate split directory
                out_dir = os.path.join(DATA_DIR, split, '0_safe')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{chunk_id}.pt")
                torch.save(features, out_path)
                processed += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {flac_path}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(flac_files)} files, {processed} chunks saved, "
                  f"{skipped_silence} silence, {skipped_short} short, {errors} errors",
                  flush=True)

    print(f"\n  Done. Processed: {processed}, Silence: {skipped_silence}, "
          f"Short: {skipped_short}, Errors: {errors}")
    return processed


def verify_speech_data():
    """Count speech .pt files per split."""
    print("\n  Speech data verification:")
    total = 0
    for split in ['train', 'val', 'test']:
        safe_dir = os.path.join(DATA_DIR, split, '0_safe')
        if not os.path.exists(safe_dir):
            continue
        speech_files = [f for f in os.listdir(safe_dir) if f.startswith('speech_')]
        count = len(speech_files)
        total += count
        print(f"    {split}/0_safe: {count} speech files")
    print(f"    Total: {total} speech files")
    return total


def main():
    parser = argparse.ArgumentParser(description='Download speech data for safe class')
    parser.add_argument('--max-samples', type=int, default=20000,
                        help='Maximum speech chunks to generate (default: 20000)')
    parser.add_argument('--include-360', action='store_true',
                        help='Also download clean-360 (23GB, 921 speakers)')
    args = parser.parse_args()

    print("=" * 60)
    print(" Download High-Quality Speech Data")
    print("=" * 60)

    # Download LibriSpeech clean-100
    print("\n1. LibriSpeech clean-100 (~6GB, 251 speakers)")
    extracted = download_librispeech(LIBRISPEECH_URL_100, SPEECH_RAW_DIR)

    # Process into PCEN spectrograms
    print("\n2. Processing speech files into PCEN spectrograms...")
    n = process_speech_files(extracted, max_samples=args.max_samples)

    if n < args.max_samples and args.include_360:
        print(f"\n3. Need more data ({n}/{args.max_samples}). Downloading clean-360...")
        extracted_360 = download_librispeech(LIBRISPEECH_URL_360, SPEECH_RAW_DIR)
        remaining = args.max_samples - n
        process_speech_files(extracted_360, max_samples=remaining)

    # Verify
    verify_speech_data()

    print("\n  Speech data ready. Retrain with:")
    print("  PYTHONPATH=. python safecommute/pipeline/train.py "
          "--focal --cosine --strong-aug --gamma 0.5 --noise-inject --seed 42")


if __name__ == "__main__":
    main()
