"""
Reconstruction-quality metrics with bootstrap 95 % CIs, for the privacy
attack harness. Each metric maps a list of (original_audio, reconstructed_audio)
pairs to a per-pair score; we aggregate with the mean and a percentile
bootstrap.

Metrics
-------
- ASR word-error rate (Whisper-tiny). Original-vs-reconstructed transcripts.
  Lower = adversary recovers more intelligible speech.
- Speaker-verification cosine (SpeechBrain ECAPA-TDNN, voxceleb).
  Reconstructed vs original embedding similarity. Compared against a
  chance baseline (cosine of N random clip pairs). Higher = adversary
  recovers speaker identity.
- PESQ wideband (16 kHz). Standardised perceptual speech quality. 1.0 to
  4.5; higher = more intelligible.
- STOI (16 kHz). Short-time objective intelligibility, 0.0 to 1.0;
  higher = more intelligible.

Lazy-loads the heavyweight models on first call. Subsequent calls reuse
the cached models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from safecommute.constants import SAMPLE_RATE


SPEECHBRAIN_CACHE = Path(
    os.environ.get(
        "SAFECOMMUTE_SPEECHBRAIN_CACHE",
        str(Path.home() / ".cache" / "safecommute-ai" / "speechbrain"),
    )
)


# ───────────────────────── bootstrap ─────────────────────────

def bootstrap_ci(values: Sequence[float],
                 n_resamples: int = 10_000,
                 alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (mean, lower_q, upper_q)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)


# ───────────────────────── Whisper WER ───────────────────────

@lru_cache(maxsize=1)
def _load_whisper():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "openai/whisper-tiny.en"
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(name).to(device)
    model.eval()
    return processor, model, device


def transcribe(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Transcribe a single 16 kHz mono float32 audio array with Whisper-tiny."""
    processor, model, device = _load_whisper()
    audio = audio.astype(np.float32)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(device)
    with torch.no_grad():
        out = model.generate(feats, max_new_tokens=128)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def wer_pairs(originals: Sequence[np.ndarray],
              reconstructions: Sequence[np.ndarray],
              ref_transcripts: Sequence[str] | None = None,
              sr: int = SAMPLE_RATE) -> tuple[list[float], list[dict]]:
    """
    Per-clip WER on Whisper-tiny.en transcripts of (a) the reconstructed
    audio vs (b) a reference transcript. If `ref_transcripts` is given,
    it is used as the reference (LibriSpeech ground truth). Otherwise we
    Whisper-transcribe the originals and use those as references.

    Returns (per_clip_wer, examples) where examples is a list of dicts
    with the reference and hypothesis strings, useful for the report.
    """
    from jiwer import wer

    if ref_transcripts is None:
        ref_transcripts = [transcribe(a, sr) for a in originals]

    per_clip = []
    examples = []
    for orig, recon, ref in zip(originals, reconstructions, ref_transcripts):
        hyp = transcribe(recon, sr)
        try:
            w = float(wer(ref, hyp if hyp else " "))
        except Exception:
            w = 1.0
        # jiwer's WER can exceed 1.0 due to insertions; cap at 1.0 so the
        # metric is on a stable 0..1 scale.
        w = min(w, 1.0)
        per_clip.append(w)
        examples.append({"reference": ref, "hypothesis": hyp, "wer": w})
    return per_clip, examples


# ───────────────────────── speaker cosine ────────────────────

@lru_cache(maxsize=1)
def _load_speaker_encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(SPEECHBRAIN_CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )
    return enc, device


def _embed(enc, audio: np.ndarray, device: str) -> np.ndarray:
    a = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = enc.encode_batch(a).squeeze().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-12)


def speaker_cosine_pairs(originals: Sequence[np.ndarray],
                          reconstructions: Sequence[np.ndarray],
                          chance_seed: int = 0) -> tuple[list[float], list[float]]:
    """
    Returns (per_pair_cosine, chance_cosines).

    `per_pair_cosine[i]` is cos(orig[i], recon[i]); recovery score.
    `chance_cosines` are N cosines from random non-matching pairs of
    originals (orig[i], orig[shuffled[i]]); the random-pair baseline.
    """
    enc, device = _load_speaker_encoder()
    orig_embs = [_embed(enc, a, device) for a in originals]
    recon_embs = [_embed(enc, a, device) for a in reconstructions]
    per_pair = [float(np.dot(o, r)) for o, r in zip(orig_embs, recon_embs)]

    rng = np.random.default_rng(chance_seed)
    perm = rng.permutation(len(orig_embs))
    # Avoid trivial self-pairs
    for i in range(len(perm)):
        if perm[i] == i:
            perm[i] = (i + 1) % len(perm)
    chance = [float(np.dot(orig_embs[i], orig_embs[perm[i]])) for i in range(len(orig_embs))]
    return per_pair, chance


# ───────────────────────── PESQ + STOI ───────────────────────

def pesq_pairs(originals: Sequence[np.ndarray],
               reconstructions: Sequence[np.ndarray],
               sr: int = SAMPLE_RATE) -> list[float]:
    """PESQ wideband (16 kHz). Catches per-clip failures and emits NaN."""
    from pesq import pesq
    out = []
    for o, r in zip(originals, reconstructions):
        try:
            score = float(pesq(sr, o.astype(np.float32), r.astype(np.float32), "wb"))
        except Exception:
            score = float("nan")
        out.append(score)
    return out


def stoi_pairs(originals: Sequence[np.ndarray],
               reconstructions: Sequence[np.ndarray],
               sr: int = SAMPLE_RATE) -> list[float]:
    from pystoi import stoi
    out = []
    for o, r in zip(originals, reconstructions):
        try:
            score = float(stoi(o.astype(np.float32), r.astype(np.float32),
                                sr, extended=False))
        except Exception:
            score = float("nan")
        out.append(score)
    return out


# ───────────────────────── aggregator ────────────────────────

@dataclass
class MetricSummary:
    name: str
    n: int
    mean: float
    ci95_lo: float
    ci95_hi: float

    def as_dict(self) -> dict:
        return {
            "metric": self.name,
            "n": self.n,
            "mean": self.mean,
            "ci95_lo": self.ci95_lo,
            "ci95_hi": self.ci95_hi,
        }


def summarise(name: str, values: Sequence[float], drop_nan: bool = True) -> MetricSummary:
    arr = np.asarray(values, dtype=np.float64)
    if drop_nan:
        arr = arr[~np.isnan(arr)]
    mean, lo, hi = bootstrap_ci(arr.tolist())
    return MetricSummary(name=name, n=int(arr.size), mean=mean, ci95_lo=lo, ci95_hi=hi)
