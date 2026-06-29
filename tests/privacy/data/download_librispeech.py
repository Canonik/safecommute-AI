"""
Download LibriSpeech dev-clean from openslr.org and slice it to 200
3-second 16 kHz clips that feed the privacy attack evaluation.

Choices:
  - dev-clean is the standard read-speech benchmark slice. Clean audio is
    the worst case for the privacy claim: if dirty/noisy speech survives
    the PCEN pipeline more easily than clean speech, we want to know, but
    clean speech is the strongest baseline.
  - 3 s clips match the classifier's input window. Clips shorter than 3 s
    are skipped; longer clips are truncated from the start.
  - n=200 keeps bootstrap 95 % CIs under roughly 5 pp half-width on WER
    while staying under 10 minutes total audio so the full attack eval fits
    in tens of minutes on the dev box.
  - We sample deterministically by hashing (speaker_id, chapter_id, id);
    sort the candidate set by that hash and take the first 200. Re-runs
    produce identical clip sets.

Source: https://www.openslr.org/resources/12/dev-clean.tar.gz (337 MB).
Idempotent: the tarball, the extracted tree, and the sliced clips are
each checked before re-creating.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from safecommute.constants import SAMPLE_RATE


N_CLIPS = 200
CLIP_DURATION_SEC = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * CLIP_DURATION_SEC)

DATA_ROOT = Path(__file__).resolve().parent
TAR_PATH = DATA_ROOT / "dev-clean.tar.gz"
EXTRACT_ROOT = DATA_ROOT / "LibriSpeech"
OUT_DIR = DATA_ROOT / "librispeech_devclean_3s"
URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


def _hash_key(speaker_id: str, chapter_id: str, sample_id: str) -> int:
    raw = f"{speaker_id}|{chapter_id}|{sample_id}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _download_tarball():
    if TAR_PATH.exists() and TAR_PATH.stat().st_size > 100_000_000:
        return
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp, open(TAR_PATH, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        chunk = 1024 * 1024
        read = 0
        next_pct = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = int(100 * read / total)
                if pct >= next_pct:
                    print(f"  {pct:3d}%  ({read / 1e6:.1f} / {total / 1e6:.1f} MB)")
                    next_pct = pct + 10
    print(f"  saved {TAR_PATH.stat().st_size / 1e6:.1f} MB")


def _extract():
    target_inner = EXTRACT_ROOT / "dev-clean"
    if target_inner.exists():
        return
    print(f"extracting {TAR_PATH.name} ...")
    with tarfile.open(str(TAR_PATH), "r:gz") as tar:
        tar.extractall(path=DATA_ROOT)
    print(f"  extracted to {EXTRACT_ROOT}")


def _read_transcripts(chapter_dir: Path) -> dict:
    transcripts = {}
    for trans_file in chapter_dir.glob("*.trans.txt"):
        for line in trans_file.read_text().splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            transcripts[utt_id] = text.strip()
    return transcripts


def _collect_candidates():
    candidates = []
    dev_clean = EXTRACT_ROOT / "dev-clean"
    for spk_dir in sorted(dev_clean.iterdir()):
        if not spk_dir.is_dir():
            continue
        for chap_dir in sorted(spk_dir.iterdir()):
            if not chap_dir.is_dir():
                continue
            transcripts = _read_transcripts(chap_dir)
            for flac in sorted(chap_dir.glob("*.flac")):
                utt_id = flac.stem  # e.g. "1272-128104-0000"
                if utt_id not in transcripts:
                    continue
                info = sf.info(str(flac))
                if info.samplerate != SAMPLE_RATE or info.frames < TARGET_LENGTH:
                    continue
                speaker_id, chapter_id, sample_id = utt_id.split("-")
                candidates.append({
                    "clip_id": utt_id,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "flac": flac,
                    "transcript": transcripts[utt_id],
                })
    return candidates


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        all_present = all(
            (OUT_DIR / f"{c['clip_id']}.wav").exists() for c in manifest.get("clips", [])
        )
        if all_present and len(manifest.get("clips", [])) >= N_CLIPS:
            print(f"already have {len(manifest['clips'])} clips at {OUT_DIR}; skipping")
            return

    _download_tarball()
    _extract()

    candidates = _collect_candidates()
    print(f"{len(candidates)} dev-clean clips pass the >= 3 s, 16 kHz filter")
    if len(candidates) < N_CLIPS:
        raise RuntimeError(f"only {len(candidates)} clips available; need {N_CLIPS}")

    candidates.sort(key=lambda c: _hash_key(c["speaker_id"], c["chapter_id"], c["clip_id"]))
    selected = candidates[:N_CLIPS]

    clips_meta = []
    for c in selected:
        audio, sr = sf.read(str(c["flac"]), dtype="float32")
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = np.asarray(audio[:TARGET_LENGTH], dtype=np.float32)
        out_wav = OUT_DIR / f"{c['clip_id']}.wav"
        sf.write(str(out_wav), audio, SAMPLE_RATE)
        clips_meta.append({
            "clip_id": c["clip_id"],
            "speaker_id": c["speaker_id"],
            "chapter_id": c["chapter_id"],
            "transcript": c["transcript"],
        })

    manifest = {
        "corpus": "librispeech_devclean_3s",
        "source": URL,
        "sample_rate": SAMPLE_RATE,
        "clip_duration_sec": CLIP_DURATION_SEC,
        "target_length_samples": TARGET_LENGTH,
        "n_clips": len(clips_meta),
        "clips": clips_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(clips_meta)} clips and manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
