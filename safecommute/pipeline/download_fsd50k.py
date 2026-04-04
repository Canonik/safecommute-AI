"""
Download FSD50K clips for SafeCommute AI (fallback if AudioSet yields too few clips).

FSD50K (Fonseca et al., 2022) is hosted on Zenodo as direct ZIP downloads —
no YouTube dependency, no rate limiting, no deleted videos.

Downloads the dev and eval audio + ground truth CSVs, then copies clips
matching our threat/safe categories into raw_data/fsd50k/{threat|safe}/.

Usage:
    PYTHONPATH=. python safecommute/pipeline/download_fsd50k.py
    PYTHONPATH=. python safecommute/pipeline/download_fsd50k.py --skip-download  # only reorganize
"""

import os
import sys
import csv
import shutil
import zipfile
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import RAW_DIR

FSD50K_BASE = os.path.join(RAW_DIR, 'fsd50k')
FSD50K_DOWNLOAD = os.path.join(FSD50K_BASE, 'download')

# Zenodo direct download URLs for FSD50K
# https://zenodo.org/record/4060432
ZENODO_FILES = {
    'FSD50K.dev_audio.zip': 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip',
    'FSD50K.eval_audio.zip': 'https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip',
    'FSD50K.ground_truth.zip': 'https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip',
}

# FSD50K label mapping to our categories
# FSD50K uses AudioSet ontology labels but with underscores
THREAT_LABELS = {
    'Screaming':             'screaming',
    'Shatter':               'glass_breaking',
    'Gunshot_and_gunfire':   'gunshot',
    'Explosion':             'explosion',
}

SAFE_LABELS = {
    'Laughter':  'laughter',
    'Crowd':     'crowd',
    'Speech':    'speech',
    'Music':     'music',
    'Applause':  'applause',
}


def download_zenodo_files():
    """Download FSD50K ZIP files from Zenodo."""
    import requests

    os.makedirs(FSD50K_DOWNLOAD, exist_ok=True)

    for fname, url in ZENODO_FILES.items():
        path = os.path.join(FSD50K_DOWNLOAD, fname)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  {fname}: already present ({os.path.getsize(path) // (1024*1024)} MB)")
            continue

        print(f"  {fname}: downloading from Zenodo...")
        try:
            r = requests.get(url, timeout=1800, stream=True)
            r.raise_for_status()
            total = 0
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    total += len(chunk)
                    if total % (500 * 1024 * 1024) == 0:
                        print(f"    Downloaded {total // (1024*1024)} MB...")
            print(f"  {fname}: done ({total // (1024*1024)} MB)")
        except Exception as e:
            print(f"  {fname}: FAILED ({e})")
            if os.path.exists(path):
                os.remove(path)


def extract_zips():
    """Extract downloaded ZIP files."""
    for fname in ZENODO_FILES:
        path = os.path.join(FSD50K_DOWNLOAD, fname)
        if not os.path.exists(path):
            continue

        # Check if already extracted (look for a marker)
        extract_dir = os.path.join(FSD50K_DOWNLOAD, fname.replace('.zip', ''))
        if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
            print(f"  {fname}: already extracted")
            continue

        print(f"  {fname}: extracting...")
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(FSD50K_DOWNLOAD)
            print(f"  {fname}: extracted")
        except Exception as e:
            print(f"  {fname}: extraction FAILED ({e})")


def find_ground_truth():
    """Find and parse FSD50K ground truth CSV files."""
    # Ground truth files are in FSD50K.ground_truth/
    gt_dir = os.path.join(FSD50K_DOWNLOAD, 'FSD50K.ground_truth')
    if not os.path.exists(gt_dir):
        # Try without the dot prefix
        for d in os.listdir(FSD50K_DOWNLOAD):
            if 'ground_truth' in d.lower() and os.path.isdir(os.path.join(FSD50K_DOWNLOAD, d)):
                gt_dir = os.path.join(FSD50K_DOWNLOAD, d)
                break

    clips = {}  # fname_no_ext -> set of labels

    for csv_name in ['dev.csv', 'eval.csv']:
        csv_path = os.path.join(gt_dir, csv_name)
        if not os.path.exists(csv_path):
            # Try alternative paths
            for root, dirs, files in os.walk(FSD50K_DOWNLOAD):
                for f in files:
                    if f == csv_name:
                        csv_path = os.path.join(root, f)
                        break

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_name} not found")
            continue

        print(f"  Parsing {csv_path}...")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                fname = row[0].strip()
                labels = row[1].strip().split(',') if len(row) > 1 else []
                clips[fname] = set(l.strip() for l in labels)

    return clips


def find_audio_files():
    """Find all extracted audio files (.wav) from FSD50K."""
    audio_files = {}  # fname_no_ext -> full_path

    for subdir in ['FSD50K.dev_audio', 'FSD50K.eval_audio']:
        audio_dir = os.path.join(FSD50K_DOWNLOAD, subdir)
        if not os.path.exists(audio_dir):
            # Search for it
            for root, dirs, files in os.walk(FSD50K_DOWNLOAD):
                if subdir in dirs:
                    audio_dir = os.path.join(root, subdir)
                    break

        if not os.path.exists(audio_dir):
            continue

        for fname in os.listdir(audio_dir):
            if fname.endswith('.wav'):
                key = fname.replace('.wav', '')
                audio_files[key] = os.path.join(audio_dir, fname)

    return audio_files


def organize_clips(clips, audio_files):
    """Copy matching clips into our category structure."""
    all_labels = {**THREAT_LABELS, **SAFE_LABELS}
    summary = {}

    for fsd_label, our_name in all_labels.items():
        group = 'threat' if fsd_label in THREAT_LABELS else 'safe'
        out_dir = os.path.join(FSD50K_BASE, group, our_name)
        os.makedirs(out_dir, exist_ok=True)

        count = 0
        for fname, labels in clips.items():
            if fsd_label not in labels:
                continue
            if fname not in audio_files:
                continue

            src = audio_files[fname]
            dst = os.path.join(out_dir, f"fsd_{fname}.wav")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            count += 1

        summary[our_name] = {'group': group, 'count': count}
        print(f"  [{group}] {our_name}: {count} clips")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Download FSD50K clips for SafeCommute AI (AudioSet fallback)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, only reorganize already-extracted files')
    args = parser.parse_args()

    print("=" * 60)
    print(" SafeCommute AI — FSD50K Downloader (AudioSet fallback)")
    print("=" * 60)

    # Step 1: Download ZIPs
    if not args.skip_download:
        print("\n" + "─" * 60)
        print(" Step 1: Download from Zenodo")
        print("─" * 60)
        download_zenodo_files()

        print("\n" + "─" * 60)
        print(" Step 2: Extract ZIPs")
        print("─" * 60)
        extract_zips()
    else:
        print("\n  Skipping download (--skip-download)")

    # Step 3: Parse ground truth
    print("\n" + "─" * 60)
    print(" Step 3: Parse ground truth labels")
    print("─" * 60)
    clips = find_ground_truth()
    print(f"  Total annotated clips: {len(clips)}")

    if not clips:
        print("  ERROR: No ground truth found. Check extraction.")
        return

    # Step 4: Find audio files
    audio_files = find_audio_files()
    print(f"  Total audio files found: {len(audio_files)}")

    if not audio_files:
        print("  ERROR: No audio files found. Check extraction.")
        return

    # Step 5: Organize
    print("\n" + "─" * 60)
    print(" Step 4: Copy matching clips to output directories")
    print("─" * 60)
    summary = organize_clips(clips, audio_files)

    # Summary
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    threat_total = sum(s['count'] for s in summary.values() if s['group'] == 'threat')
    safe_total = sum(s['count'] for s in summary.values() if s['group'] == 'safe')
    print(f"  Threat clips: {threat_total}")
    print(f"  Safe clips:   {safe_total}")
    print(f"  Output:       {FSD50K_BASE}/{{threat|safe}}/{{category}}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
