"""
Download AudioSet strongly-labeled audio clips for SafeCommute AI.

AudioSet (Gemmeke et al., 2017) is a large-scale dataset of 10-second
YouTube audio clips annotated with 527 sound event labels. We use the
*strongly-labeled* subset where each clip has verified start/end times
for each label — this avoids noisy weak labels.

We download specific threat and hard-negative categories via yt-dlp,
converting to 16kHz mono WAV on the fly.

Expect ~30-40% failure rate (deleted/private/geo-blocked YouTube videos).

Usage:
    PYTHONPATH=. python safecommute/pipeline/download_audioset.py
    PYTHONPATH=. python safecommute/pipeline/download_audioset.py --max-per-category 100 --dry-run
    PYTHONPATH=. python safecommute/pipeline/download_audioset.py --threat-only --sleep 3
"""

import os
import sys
import csv
import time
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.constants import SAMPLE_RATE, RAW_DIR

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Universal threat sounds. These categories were chosen because they
# represent the core acoustic signatures of physical danger in public spaces.
# EXCLUDED categories and rationale:
#   - Siren (/m/03kmc9): contextual confound — sirens indicate emergency
#     response, not an active threat. Including them causes the model to
#     alert on police/ambulance presence, which is a false positive.
#   - Crying/sobbing (/m/0463cq4): emotional distress, not physical threat.
#     Crying is common in public transit (babies) and would cause false alarms.
#   - Fighting (/m/07s0s5r): AudioSet's "Fighting" is a topic label with
#     <20 strongly-labeled clips — too few for reliable training.
THREAT_CATEGORIES = {
    "screaming":      "/m/03qc9zr",
    "shout":          "/m/07sr1lc",
    "yell":           "/m/07r660_",
    "gunshot":        "/m/032s66",
    "explosion":      "/m/014zdl",
    "glass_breaking": "/m/07q0yl5",
}

# Layer 2 — Universal hard negatives (safe class). These are sounds that are
# acoustically similar to threats (loud, abrupt, high-energy) but represent
# normal public-space activity. Including them as safe training data forces
# the model to learn fine-grained distinctions (e.g., cheering vs. screaming,
# applause vs. gunshot reverb).
SAFE_CATEGORIES = {
    "laughter":  "/m/01j3sz",
    "crowd":     "/m/03qtwd",
    "speech":    "/m/09x0r",
    "music":     "/m/04rlf",
    "applause":  "/m/0fx80y",
    "cheering":  "/m/03w41f",
    "singing":   "/m/015lz1",
}

AUDIOSET_BASE = os.path.join(RAW_DIR, 'audioset')
METADATA_DIR = os.path.join(AUDIOSET_BASE, 'metadata')
FAILURE_LOG = os.path.join(AUDIOSET_BASE, 'failures.log')

SEGMENT_CSVS = {
    'eval_segments.csv':
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
    'balanced_train_segments.csv':
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
    'unbalanced_train_segments.csv':
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv',
}


# ─────────────────────────────────────────────────────────────────────────────
# METADATA DOWNLOAD & PARSING
# ─────────────────────────────────────────────────────────────────────────────
def download_metadata():
    """Download AudioSet CSV metadata files if not already present."""
    os.makedirs(METADATA_DIR, exist_ok=True)
    import requests

    for fname, url in SEGMENT_CSVS.items():
        path = os.path.join(METADATA_DIR, fname)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  {fname}: already present ({os.path.getsize(path) // 1024} KB)")
            continue
        print(f"  {fname}: downloading...")
        try:
            r = requests.get(url, timeout=600, stream=True)
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"  {fname}: done ({os.path.getsize(path) // 1024} KB)")
        except Exception as e:
            print(f"  {fname}: FAILED ({e})")


def parse_segments_for_categories(target_ids, max_per_category):
    """
    Parse all AudioSet segment CSVs and return clips matching target category IDs.

    AudioSet CSV format (after comment lines starting with #):
      YTID, start_seconds, end_seconds, positive_labels
    where positive_labels is a quoted comma-separated list of ontology MIDs
    (e.g., "/m/03qc9zr,/m/07sr1lc"). Fields are separated by ", " (comma-space).

    We parse all three segment files (eval, balanced_train, unbalanced_train)
    in order. The unbalanced set is ~100MB so it is streamed line-by-line.
    Early exit occurs once all categories reach max_per_category.

    Returns: dict[category_mid] -> list of (video_id, start_sec, end_sec)
    """
    target_set = set(target_ids.values())
    results = {mid: [] for mid in target_set}
    counts = {mid: 0 for mid in target_set}

    for fname in SEGMENT_CSVS:
        path = os.path.join(METADATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  Warning: {fname} not found, skipping")
            continue

        print(f"  Parsing {fname}...")
        with open(path, 'r') as f:
            for line in f:
                # Skip comment lines (start with #) and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Format: YTID, start_seconds, end_seconds, positive_labels
                # Labels field is space-prefixed and quoted:
                #   "YTID", 30.000, 40.000, "/m/id1,/m/id2"
                # But sometimes no quotes. Handle both.
                try:
                    # Split on comma but respect that labels field has commas inside quotes
                    parts = line.split(', ', 3)
                    if len(parts) < 4:
                        continue
                    video_id = parts[0].strip().strip('"')
                    start = float(parts[1].strip())
                    end = float(parts[2].strip())
                    labels_raw = parts[3].strip().strip('"')
                    labels = [l.strip() for l in labels_raw.split(',')]
                except (ValueError, IndexError):
                    continue

                # Check if any of our target categories are in this clip's labels
                for mid in target_set:
                    if mid in labels and counts[mid] < max_per_category:
                        results[mid].append((video_id, start, end))
                        counts[mid] += 1

        # Early exit if all categories are full
        if all(c >= max_per_category for c in counts.values()):
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_clip(video_id, start, end, output_path, sleep_sec):
    """
    Download a single AudioSet segment via yt-dlp.

    Uses --download-sections to grab only the labeled time range (typically
    10 seconds), converting on-the-fly to 16kHz mono WAV. The sleep between
    downloads avoids YouTube rate-limiting.

    Expect ~30-40% failure rate due to deleted/private/geo-blocked videos.
    Failures are logged to failures.log for later analysis.
    """
    if os.path.exists(output_path):
        return 'skipped'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # yt-dlp command: extract audio, convert to 16kHz mono WAV, download only the segment
    cmd = [
        'yt-dlp',
        '-x', '--audio-format', 'wav',
        '--postprocessor-args', f'ffmpeg:-ar {SAMPLE_RATE} -ac 1',
        '--download-sections', f'*{start:.1f}-{end:.1f}',
        '-o', output_path,
        '--no-playlist',
        '--quiet', '--no-warnings',
        f'https://www.youtube.com/watch?v={video_id}',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_path):
            time.sleep(sleep_sec)
            return 'success'
        else:
            # Log failure
            with open(FAILURE_LOG, 'a') as f:
                f.write(f"{video_id}\t{start}\t{end}\t{result.stderr[:200]}\n")
            time.sleep(sleep_sec)
            return 'failed'
    except subprocess.TimeoutExpired:
        with open(FAILURE_LOG, 'a') as f:
            f.write(f"{video_id}\t{start}\t{end}\ttimeout\n")
        time.sleep(sleep_sec)
        return 'failed'
    except Exception as e:
        with open(FAILURE_LOG, 'a') as f:
            f.write(f"{video_id}\t{start}\t{end}\t{e}\n")
        return 'failed'


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Download AudioSet strongly-labeled clips for SafeCommute AI')
    parser.add_argument('--max-per-category', type=int, default=300,
                        help='Max clips to attempt per category (default: 300)')
    parser.add_argument('--categories', type=str, default=None,
                        help='Comma-separated category names to download (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print what would be downloaded')
    parser.add_argument('--sleep', type=float, default=1.5,
                        help='Seconds between downloads (default: 1.5)')
    parser.add_argument('--threat-only', action='store_true',
                        help='Only download threat categories')
    parser.add_argument('--safe-only', action='store_true',
                        help='Only download safe/hard-negative categories')
    args = parser.parse_args()

    print("=" * 60)
    print(" SafeCommute AI — AudioSet Downloader")
    print("=" * 60)

    # Determine which categories to download
    categories = {}
    if not args.safe_only:
        for name, mid in THREAT_CATEGORIES.items():
            categories[name] = ('threat', mid)
    if not args.threat_only:
        for name, mid in SAFE_CATEGORIES.items():
            categories[name] = ('safe', mid)

    if args.categories:
        filter_names = set(args.categories.split(','))
        categories = {k: v for k, v in categories.items() if k in filter_names}

    if not categories:
        print("No categories selected. Exiting.")
        return

    print(f"\nCategories: {len(categories)}")
    for name, (group, mid) in categories.items():
        print(f"  [{group}] {name}: {mid}")
    print(f"Max per category: {args.max_per_category}")
    print(f"Sleep between downloads: {args.sleep}s")

    # Step 1: Download metadata CSVs
    print("\n" + "─" * 60)
    print(" Step 1: Download AudioSet metadata")
    print("─" * 60)
    download_metadata()

    # Step 2: Parse segments for our categories
    print("\n" + "─" * 60)
    print(" Step 2: Parse segment CSVs for target categories")
    print("─" * 60)
    all_target_ids = {name: mid for name, (_, mid) in categories.items()}
    clip_map = parse_segments_for_categories(all_target_ids, args.max_per_category)

    # Invert mid->name for output paths
    mid_to_name = {mid: name for name, mid in all_target_ids.items()}
    mid_to_group = {mid: group for _, (group, mid) in categories.items()}

    total_found = sum(len(clips) for clips in clip_map.values())
    print(f"\n  Total clips found: {total_found}")
    for mid, clips in clip_map.items():
        name = mid_to_name.get(mid, mid)
        print(f"    {name}: {len(clips)} clips")

    if args.dry_run:
        print("\n[DRY RUN] Would download the above. Exiting.")
        return

    # Step 3: Download audio
    print("\n" + "─" * 60)
    print(" Step 3: Download audio clips via yt-dlp")
    print("─" * 60)

    os.makedirs(AUDIOSET_BASE, exist_ok=True)

    summary = {}  # category_name -> {success, failed, skipped}

    for mid, clips in clip_map.items():
        name = mid_to_name.get(mid, mid)
        group = mid_to_group.get(mid, 'unknown')
        out_dir = os.path.join(AUDIOSET_BASE, group, name)
        os.makedirs(out_dir, exist_ok=True)

        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total': len(clips)}
        print(f"\n  [{group}] {name}: {len(clips)} clips to process")

        for i, (vid, start, end) in enumerate(clips):
            out_path = os.path.join(out_dir, f"{vid}_{start:.0f}_{end:.0f}.wav")
            result = download_clip(vid, start, end, out_path, args.sleep)
            stats[result] += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(clips):
                print(f"    [{i+1}/{len(clips)}] "
                      f"ok={stats['success']} fail={stats['failed']} skip={stats['skipped']}")

        summary[name] = stats

    # Step 4: Summary
    print("\n" + "=" * 60)
    print(" Download Summary")
    print("=" * 60)
    print(f"  {'Category':<18} {'Group':<8} {'Found':>6} {'OK':>6} {'Fail':>6} {'Skip':>6}")
    print("  " + "─" * 54)

    total_ok = total_fail = total_skip = 0
    for name, stats in summary.items():
        group = mid_to_group.get(all_target_ids[name], '?')
        print(f"  {name:<18} {group:<8} {stats['total']:>6} "
              f"{stats['success']:>6} {stats['failed']:>6} {stats['skipped']:>6}")
        total_ok += stats['success']
        total_fail += stats['failed']
        total_skip += stats['skipped']

    print("  " + "─" * 54)
    print(f"  {'TOTAL':<18} {'':8} {total_ok + total_fail + total_skip:>6} "
          f"{total_ok:>6} {total_fail:>6} {total_skip:>6}")

    # Warning if too few threat clips
    threat_ok = sum(
        s['success'] + s['skipped']
        for name, s in summary.items()
        if all_target_ids[name] in [mid for mid in THREAT_CATEGORIES.values()]
    )
    if threat_ok < 500:
        print(f"\n  WARNING: Only {threat_ok} threat clips obtained.")
        print("  Consider running: PYTHONPATH=. python safecommute/pipeline/download_fsd50k.py")

    print("\nDone.")


if __name__ == "__main__":
    main()
