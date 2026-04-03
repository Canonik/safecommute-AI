"""
Standalone dataset downloader for SafeCommute AI.
Downloads CREMA-D (via HuggingFace), TESS (via Kaggle/Dataverse),
SAVEE (via Kaggle), ESC-50 (via GitHub), and verifies RAVDESS.

Usage:
    PYTHONPATH=. python v_3/download_datasets.py
"""

import os
import sys
import json
import zipfile
import io
import shutil

import numpy as np
import requests
import soundfile as sf
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import RAW_DIR, SAMPLE_RATE

os.makedirs(os.path.join(RAW_DIR, 'ravdess'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'tess'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'cremad'), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'savee'), exist_ok=True)


def count_real_wavs(directory):
    """Count .wav files that are actual audio (not LFS pointers)."""
    count = 0
    for f in os.listdir(directory):
        if f.endswith('.wav'):
            fpath = os.path.join(directory, f)
            if os.path.getsize(fpath) > 1000:
                count += 1
    return count


def download_ravdess():
    target_dir = os.path.join(RAW_DIR, 'ravdess')
    # Count actual wav files recursively
    wav_count = sum(1 for r, _, fs in os.walk(target_dir) for f in fs if f.endswith('.wav'))
    if wav_count > 50:
        print(f"  RAVDESS: already present ({wav_count} wav files)")
        return True
    print("  RAVDESS: downloading (~200MB)...")
    try:
        r = requests.get(
            "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
            timeout=300)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(target_dir)
        print("  RAVDESS: done")
        return True
    except Exception as e:
        print(f"  RAVDESS: FAILED ({e})")
        return False


def download_cremad():
    """Download CREMA-D via HuggingFace datasets library."""
    target_dir = os.path.join(RAW_DIR, 'cremad')
    real_count = count_real_wavs(target_dir)
    if real_count > 100:
        print(f"  CREMA-D: already present ({real_count} wav files)")
        return True

    # Clean up any LFS pointers
    for f in os.listdir(target_dir):
        fpath = os.path.join(target_dir, f)
        if f.endswith('.wav') and os.path.getsize(fpath) < 1000:
            os.remove(fpath)

    print("  CREMA-D: downloading via HuggingFace (~600MB)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("myleslinder/crema-d", split="train")
        saved = 0
        for item in ds:
            audio = item.get('audio')
            if audio is None:
                continue
            # Build filename from metadata
            actor = item.get('actor_id', saved)
            sentence = item.get('sentence', 'UNK')
            emotion = item.get('emotion', 'UNK')
            intensity = item.get('emotion_intensity', 'XX')
            wav_name = f"{actor}_{sentence}_{emotion}_{intensity}.wav"
            out_path = os.path.join(target_dir, wav_name)
            if not os.path.exists(out_path):
                arr = np.array(audio['array'], dtype=np.float32)
                sr = audio.get('sampling_rate', 16000)
                sf.write(out_path, arr, sr)
                saved += 1
                if saved % 500 == 0:
                    print(f"    Saved {saved} clips...")
        print(f"  CREMA-D: done ({saved} clips)")
        return True
    except Exception as e:
        print(f"  CREMA-D: FAILED ({e})")
        return False


def download_tess():
    """Download TESS from Borealis/Dataverse."""
    target_dir = os.path.join(RAW_DIR, 'tess')
    wav_count = sum(1 for r, _, fs in os.walk(target_dir) for f in fs if f.endswith('.wav'))
    if wav_count > 100:
        print(f"  TESS: already present ({wav_count} wav files)")
        return True

    print("  TESS: downloading from Kaggle/Dataverse...")
    # Try multiple sources
    urls = [
        "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            content = b''
            for chunk in r.iter_content(chunk_size=65536):
                content += chunk
            # Try to extract as zip
            try:
                zf = zipfile.ZipFile(io.BytesIO(content))
                zf.extractall(target_dir)
                wav_count = sum(1 for r2, _, fs in os.walk(target_dir) for f in fs if f.endswith('.wav'))
                if wav_count > 100:
                    print(f"  TESS: done ({wav_count} wav files)")
                    return True
            except zipfile.BadZipFile:
                continue
        except Exception as e:
            print(f"    Source failed: {e}")
            continue

    # Fallback: try HuggingFace if available
    try:
        from datasets import load_dataset
        print("    Trying HuggingFace fallback...")
        ds = load_dataset("Valdiviandres/TESS", split="train")
        saved = 0
        for item in ds:
            audio = item.get('audio')
            label = item.get('label', item.get('emotion', 'unknown'))
            if audio is None:
                continue
            wav_name = f"tess_{saved}_{label}.wav"
            out_path = os.path.join(target_dir, wav_name)
            if not os.path.exists(out_path):
                arr = np.array(audio['array'], dtype=np.float32)
                sr = audio.get('sampling_rate', 16000)
                sf.write(out_path, arr, sr)
                saved += 1
        if saved > 0:
            print(f"  TESS: done ({saved} clips from HuggingFace)")
            return True
    except Exception as e:
        print(f"  TESS: HuggingFace fallback failed ({e})")

    print("  TESS: FAILED (all sources)")
    return False


def download_savee():
    """Download SAVEE from Kaggle (if kaggle API available) or HuggingFace."""
    target_dir = os.path.join(RAW_DIR, 'savee')
    wav_count = sum(1 for r, _, fs in os.walk(target_dir) for f in fs if f.endswith('.wav'))
    if wav_count > 50:
        print(f"  SAVEE: already present ({wav_count} wav files)")
        return True

    print("  SAVEE: downloading...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Fhrozen/SAVEE", split="train")
        saved = 0
        for item in ds:
            audio = item.get('audio')
            if audio is None:
                continue
            label = item.get('label', item.get('emotion', saved))
            wav_name = f"savee_{saved}_{label}.wav"
            out_path = os.path.join(target_dir, wav_name)
            if not os.path.exists(out_path):
                arr = np.array(audio['array'], dtype=np.float32)
                sr = audio.get('sampling_rate', 16000)
                sf.write(out_path, arr, sr)
                saved += 1
        if saved > 0:
            print(f"  SAVEE: done ({saved} clips)")
            return True
    except Exception as e:
        print(f"  SAVEE: HuggingFace failed ({e})")
    print("  SAVEE: FAILED")
    return False


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
    results['RAVDESS'] = download_ravdess()
    results['CREMA-D'] = download_cremad()
    results['TESS'] = download_tess()
    results['SAVEE'] = download_savee()
    results['ESC-50'] = download_esc50()

    print("\n" + "=" * 50)
    print(" Download Summary")
    print("=" * 50)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:12}: {status}")
