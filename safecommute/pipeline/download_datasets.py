"""
Dataset downloader for SafeCommute AI.
Downloads ESC-50 environmental sound dataset from GitHub.

Usage:
    PYTHONPATH=. python safecommute/pipeline/download_datasets.py
"""

import os
import sys
import zipfile
import io

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import RAW_DIR


def download_esc50():
    """Download ESC-50 from GitHub directly."""
    base_dir = os.path.join(RAW_DIR, 'esc50')
    audio_dir = os.path.join(base_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)

    wav_count = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')]) if os.path.exists(audio_dir) else 0
    if wav_count > 100:
        print(f"  ESC-50: already present ({wav_count} wav files)")
        return True

    print("  ESC-50: downloading from GitHub (~600MB)...")
    try:
        r = requests.get(
            "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
            timeout=600, stream=True)
        r.raise_for_status()
        content = b''
        total = 0
        for chunk in r.iter_content(chunk_size=65536):
            content += chunk
            total += len(chunk)
            if total % (100 * 1024 * 1024) == 0:
                print(f"    Downloaded {total // (1024*1024)} MB...")

        zf = zipfile.ZipFile(io.BytesIO(content))
        # Extract audio files and metadata
        for name in zf.namelist():
            if '/audio/' in name and name.endswith('.wav'):
                data = zf.read(name)
                wav_name = os.path.basename(name)
                with open(os.path.join(audio_dir, wav_name), 'wb') as wf:
                    wf.write(data)
            elif name.endswith('esc50.csv'):
                data = zf.read(name)
                with open(os.path.join(base_dir, 'esc50.csv'), 'wb') as wf:
                    wf.write(data)

        wav_count = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        print(f"  ESC-50: done ({wav_count} wav files)")
        return True
    except Exception as e:
        print(f"  ESC-50: FAILED ({e})")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print(" SafeCommute AI — Dataset Downloader")
    print("=" * 50)
    print()

    results = {}
    results['ESC-50'] = download_esc50()

    print("\n" + "=" * 50)
    print(" Download Summary")
    print("=" * 50)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:12}: {status}")
