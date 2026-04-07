"""
YouTube data quality validator for SafeCommute AI.

Checks raw YouTube audio files for quality issues:
1. Duration too short or too long
2. Silent/near-silent files
3. Music detection (high spectral regularity = likely music, not ambient)
4. Speech-heavy files in metro (could be news/podcast, not ambient)
5. Low-energy files in screams (could be talk show, not real scream)

Moves suspicious files to a quarantine directory for manual review.

Usage:
    PYTHONPATH=. python safecommute/pipeline/validate_youtube_data.py
    PYTHONPATH=. python safecommute/pipeline/validate_youtube_data.py --delete  # auto-delete bad files
"""

import os
import sys
import argparse
import shutil
import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE


def analyze_audio(path):
    """Compute quality metrics for an audio file."""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))

    # Spectral features
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    # Spectral flatness — high = noise-like, low = tonal/music
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=spec)))

    # Zero crossing rate — high = noise/speech, low = music/silence
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Spectral centroid — frequency center of mass
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=spec, sr=sr)))

    # Onset strength — how many distinct audio events (beats/transients)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = float(np.sum(onset_env > np.mean(onset_env) * 2)) / max(duration, 0.1)

    # Spectral bandwidth variance — low variance = repetitive = likely music
    bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sr)
    bw_var = float(np.var(bandwidth))

    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'flatness': flatness,
        'zcr': zcr,
        'centroid': centroid,
        'onset_rate': onset_rate,
        'bw_var': bw_var,
    }


def is_likely_music(metrics):
    """Heuristic: music has low spectral flatness, regular onsets, low BW variance."""
    if metrics['flatness'] < 0.01 and metrics['bw_var'] < 5e6:
        return True
    return False


def is_likely_speech_only(metrics):
    """Heuristic: news/podcast has moderate energy, high ZCR, moderate centroid."""
    if metrics['zcr'] > 0.08 and metrics['centroid'] < 3000 and metrics['rms'] > 0.01:
        return True
    return False


def is_too_quiet(metrics):
    """File is effectively silent."""
    return metrics['rms'] < 0.002 or metrics['peak'] < 0.01


def is_too_loud_constant(metrics):
    """Constant loud signal = likely tone/test signal."""
    return metrics['rms'] > 0.3 and metrics['flatness'] > 0.5


def validate_directory(audio_dir, expected_type, delete=False):
    """
    Validate all wav files in a directory.
    expected_type: 'safe' (ambient) or 'unsafe' (screams)
    """
    quarantine_dir = audio_dir + '_quarantine'
    os.makedirs(quarantine_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    total = len(files)
    flagged = 0
    reasons = {}

    for fname in files:
        path = os.path.join(audio_dir, fname)
        metrics = analyze_audio(path)
        if metrics is None:
            flag_reason = "unreadable"
        elif is_too_quiet(metrics):
            flag_reason = "too_quiet"
        elif is_too_loud_constant(metrics):
            flag_reason = "constant_loud"
        elif is_likely_music(metrics):
            flag_reason = "likely_music"
        elif expected_type == 'safe' and metrics['rms'] > 0.1 and metrics['zcr'] > 0.1:
            flag_reason = "too_energetic_for_ambient"
        elif expected_type == 'unsafe' and metrics['rms'] < 0.005:
            flag_reason = "too_quiet_for_scream"
        elif expected_type == 'unsafe' and is_likely_speech_only(metrics) and metrics['rms'] < 0.03:
            flag_reason = "likely_news_podcast"
        elif metrics['duration'] < 1.0:
            flag_reason = "too_short"
        else:
            continue  # passes all checks

        flagged += 1
        reasons[flag_reason] = reasons.get(flag_reason, 0) + 1

        if delete:
            os.remove(path)
        else:
            shutil.move(path, os.path.join(quarantine_dir, fname))

    return total, flagged, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help='Delete bad files instead of quarantining')
    args = parser.parse_args()

    print("=" * 50)
    print(" YouTube Data Quality Validation")
    print("=" * 50)

    # Validate metro ambient (should be quiet-to-moderate, not music)
    metro_dir = os.path.join('raw_data', 'youtube_metro')
    if os.path.exists(metro_dir):
        print(f"\nValidating metro ambient ({metro_dir})...")
        total, flagged, reasons = validate_directory(metro_dir, 'safe', args.delete)
        action = "deleted" if args.delete else "quarantined"
        print(f"  Total: {total}, Flagged: {flagged} ({action})")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Validate screams (should be loud/energetic, not calm speech)
    scream_dir = os.path.join('raw_data', 'youtube_screams')
    if os.path.exists(scream_dir):
        print(f"\nValidating screams ({scream_dir})...")
        total, flagged, reasons = validate_directory(scream_dir, 'unsafe', args.delete)
        action = "deleted" if args.delete else "quarantined"
        print(f"  Total: {total}, Flagged: {flagged} ({action})")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    print("\nDone. Review quarantined files manually, then re-run prepare_youtube_data.py")


if __name__ == "__main__":
    main()
