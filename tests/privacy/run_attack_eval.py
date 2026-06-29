"""
Privacy attack harness driver.

Loops over a corpus of 16 kHz 3 s clips, runs every (recovery, attack)
combination, and emits raw_results.json plus a handful of reconstructed
sample wavs that the report can link to.

Recoveries
----------
  - pcen_oracle  : exact inverse-PCEN given the running-mean state M
                   (upper bound on reconstruction quality; "even the
                   strongest attacker can't recover speech")
  - pcen_blind   : iterative inverse-PCEN without access to M
                   (realistic attacker)
  - mel_baseline : skip PCEN entirely, use the raw mel directly
                   (ablation: how much privacy does PCEN add over plain mel?)

Attacks
-------
  - griffin_lim  : classical phase-only inverse, no pretrained model
  - hifigan      : SpeechBrain `tts-hifigan-libritts-16kHz`, off-the-shelf,
                   with the analytical 64-to-80 mel adapter

Metrics (each with bootstrap 95 % CI; see metrics.py)
  - Whisper-tiny.en WER (vs the LibriSpeech ground-truth transcript)
  - ECAPA-TDNN speaker cosine (recon vs original; compared to chance)
  - PESQ wideband (16 kHz)
  - STOI (16 kHz)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Quiet the chatty downstream libs so the driver log stays readable.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.WARNING)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


RECOVERIES = ("pcen_oracle", "pcen_blind", "mel_baseline")
ATTACKS = ("griffin_lim", "hifigan")
N_SAMPLE_WAVS_PER_CONFIG = 10

SAMPLE_RATE = 16000
TARGET_LENGTH = 48000
forward_pcen_with_state = None
inverse_pcen_oracle = None
inverse_pcen_blind = None
attack_griffin_lim = None
attack_hifigan = None
wer_pairs = None
speaker_cosine_pairs = None
pesq_pairs = None
stoi_pairs = None
summarise = None


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Scale to a known peak so PESQ/STOI don't fail on out-of-range signals."""
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return (audio * (target_peak / peak)).astype(np.float32)


def _recover_mel(pcen, M, mel_true, recovery: str) -> np.ndarray:
    if recovery == "pcen_oracle":
        return inverse_pcen_oracle(pcen, M)
    if recovery == "pcen_blind":
        return inverse_pcen_blind(pcen)
    if recovery == "mel_baseline":
        return mel_true
    raise ValueError(recovery)


def _attack(mel, attack: str) -> np.ndarray:
    if attack == "griffin_lim":
        return attack_griffin_lim.mel_to_audio(mel, target_length=TARGET_LENGTH)
    if attack == "hifigan":
        return attack_hifigan.mel_to_audio(mel, target_length=TARGET_LENGTH)
    raise ValueError(attack)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tests/privacy/data/librispeech_devclean_3s")
    parser.add_argument("--out-dir", default="tests/privacy/reports")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on number of clips (debugging)")
    parser.add_argument("--configurations", nargs="+", default=None,
                        help="explicit recovery:attack pairs to run "
                             "(default: all 6)")
    args = parser.parse_args()

    global SAMPLE_RATE, TARGET_LENGTH
    global forward_pcen_with_state, inverse_pcen_oracle, inverse_pcen_blind
    global attack_griffin_lim, attack_hifigan
    global wer_pairs, speaker_cosine_pairs, pesq_pairs, stoi_pairs, summarise

    import soundfile as sf
    from safecommute.constants import SAMPLE_RATE as _SAMPLE_RATE, TARGET_LENGTH as _TARGET_LENGTH
    from tests.privacy.inverse_pcen import (
        forward_pcen_with_state as _forward_pcen_with_state,
        inverse_pcen_oracle as _inverse_pcen_oracle,
        inverse_pcen_blind as _inverse_pcen_blind,
    )
    from tests.privacy import attack_griffin_lim as _attack_griffin_lim
    from tests.privacy import attack_hifigan as _attack_hifigan
    from tests.privacy.metrics import (
        wer_pairs as _wer_pairs,
        speaker_cosine_pairs as _speaker_cosine_pairs,
        pesq_pairs as _pesq_pairs,
        stoi_pairs as _stoi_pairs,
        summarise as _summarise,
    )

    SAMPLE_RATE = _SAMPLE_RATE
    TARGET_LENGTH = _TARGET_LENGTH
    forward_pcen_with_state = _forward_pcen_with_state
    inverse_pcen_oracle = _inverse_pcen_oracle
    inverse_pcen_blind = _inverse_pcen_blind
    attack_griffin_lim = _attack_griffin_lim
    attack_hifigan = _attack_hifigan
    wer_pairs = _wer_pairs
    speaker_cosine_pairs = _speaker_cosine_pairs
    pesq_pairs = _pesq_pairs
    stoi_pairs = _stoi_pairs
    summarise = _summarise

    corpus_dir = Path(args.corpus)
    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    clips = manifest["clips"]
    if args.limit:
        clips = clips[: args.limit]
    n = len(clips)
    print(f"loaded {n} clips from {corpus_dir}")

    if args.configurations:
        configs = [tuple(c.split(":")) for c in args.configurations]
    else:
        configs = [(r, a) for r in RECOVERIES for a in ATTACKS]
    print(f"configurations: {configs}")

    # 1. Load every clip + compute its PCEN forward (mel, M, y) once.
    print("forward pass: audio -> mel, M, pcen ...")
    originals: list[np.ndarray] = []
    mels: list[np.ndarray] = []
    pcens: list[np.ndarray] = []
    Ms: list[np.ndarray] = []
    for c in clips:
        wav_path = corpus_dir / f"{c['clip_id']}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == SAMPLE_RATE, f"clip {c['clip_id']} has sr={sr}"
        audio = np.asarray(audio[:TARGET_LENGTH], dtype=np.float32)
        if audio.shape[0] < TARGET_LENGTH:
            audio = np.pad(audio, (0, TARGET_LENGTH - audio.shape[0]))
        fwd = forward_pcen_with_state(audio)
        originals.append(audio)
        mels.append(fwd["mel"])
        pcens.append(fwd["y"])
        Ms.append(fwd["M"])

    # 2. Whisper-transcribe the ORIGINALS once. These transcripts are the
    # WER references: "what an attacker would compare their reconstruction
    # against". Using the LibriSpeech ground-truth transcript is wrong here
    # because the corpus is truncated to 3 s while the transcript covers the
    # full ~10 s clip; that inflates WER even for perfect reconstructions.
    print("transcribing originals with Whisper-tiny.en (one-time) ...")
    from tests.privacy.metrics import transcribe
    refs = [transcribe(a) for a in originals]
    librispeech_refs = [c["transcript"] for c in clips]

    # 3. For each (recovery, attack): reconstruct n waveforms, compute metrics.
    results = {
        "corpus": manifest["corpus"],
        "n_clips": n,
        "sample_rate": SAMPLE_RATE,
        "wer_reference": "whisper_tiny_en_on_originals",
        "configurations": [],
    }

    import time
    for recovery, attack in configs:
        tag = f"{recovery}__{attack}"
        print(f"\n--- {tag} ---", flush=True)
        t0 = time.time()
        recons: list[np.ndarray] = []
        for i in range(n):
            mel_hat = _recover_mel(pcens[i], Ms[i], mels[i], recovery)
            recon = _attack(mel_hat, attack)
            recon = recon[:TARGET_LENGTH]
            if recon.shape[0] < TARGET_LENGTH:
                recon = np.pad(recon, (0, TARGET_LENGTH - recon.shape[0]))
            recon = _peak_normalize(recon)
            recons.append(recon)
            if i < N_SAMPLE_WAVS_PER_CONFIG:
                sample_path = samples_dir / f"{tag}__{clips[i]['clip_id']}.wav"
                sf.write(str(sample_path), recon, SAMPLE_RATE)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  reconstructed {i + 1}/{n}  (elapsed {elapsed:.0f}s)", flush=True)

        recon_elapsed = time.time() - t0
        print(f"  reconstructions done in {recon_elapsed:.0f}s; computing metrics ...", flush=True)
        t1 = time.time()
        wer_per_clip, wer_examples = wer_pairs(originals, recons, ref_transcripts=refs)
        cos_per_pair, cos_chance = speaker_cosine_pairs(originals, recons, chance_seed=0)
        pesq_per_clip = pesq_pairs(originals, recons)
        stoi_per_clip = stoi_pairs(originals, recons)

        config_summary = {
            "recovery": recovery,
            "attack": attack,
            "metrics": {
                "wer": summarise("wer", wer_per_clip).as_dict(),
                "speaker_cosine": summarise("speaker_cosine", cos_per_pair).as_dict(),
                "speaker_cosine_chance": summarise("speaker_cosine_chance", cos_chance).as_dict(),
                "pesq": summarise("pesq", pesq_per_clip).as_dict(),
                "stoi": summarise("stoi", stoi_per_clip).as_dict(),
            },
            "examples": [
                {
                    "clip_id": clips[i]["clip_id"],
                    "reference": wer_examples[i]["reference"],
                    "hypothesis": wer_examples[i]["hypothesis"],
                    "wer": wer_examples[i]["wer"],
                    "speaker_cosine": cos_per_pair[i],
                    "pesq": pesq_per_clip[i],
                    "stoi": stoi_per_clip[i],
                    "sample_wav": f"samples/{tag}__{clips[i]['clip_id']}.wav"
                    if i < N_SAMPLE_WAVS_PER_CONFIG else None,
                }
                for i in range(min(N_SAMPLE_WAVS_PER_CONFIG, n))
            ],
        }
        results["configurations"].append(config_summary)

        m = config_summary["metrics"]
        metrics_elapsed = time.time() - t1
        print(f"    metrics done in {metrics_elapsed:.0f}s "
              f"(config total {recon_elapsed + metrics_elapsed:.0f}s)", flush=True)
        print(f"    WER mean={m['wer']['mean']:.3f}  "
              f"[{m['wer']['ci95_lo']:.3f}, {m['wer']['ci95_hi']:.3f}]", flush=True)
        print(f"    speaker cos mean={m['speaker_cosine']['mean']:.3f}  "
              f"(chance={m['speaker_cosine_chance']['mean']:.3f})", flush=True)
        print(f"    PESQ mean={m['pesq']['mean']:.3f}  "
              f"STOI mean={m['stoi']['mean']:.3f}", flush=True)

    out_path = out_dir / "raw_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
