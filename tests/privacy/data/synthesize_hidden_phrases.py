"""
Synthesize a small corpus of probe phrases for the privacy attack
hidden-phrase sub-evaluation.

The point of this corpus is to plant *known* content in audio that the
privacy attack will then try to recover. If the aggregate WER on
LibriSpeech is high, no individual keyword should survive either; this
sub-corpus makes that demonstration concrete (a paper reviewer can read
the recovered transcripts next to the planted phrases).

We synthesise with SpeechBrain's Tacotron2 + HiFi-GAN trained on
LJSpeech, then resample to 16 kHz. This pair is deliberately chosen to
be *different* from the attack vocoder (`tts-hifigan-libritts-16kHz`):
we don't want the same model to both synthesise and attempt to
reconstruct, which would short-circuit the experiment.

Output: tests/privacy/data/hidden_phrases/{idx:02d}.wav + manifest.json,
formatted to be loadable by `run_attack_eval.py --corpus`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from safecommute.constants import SAMPLE_RATE, TARGET_LENGTH


PROBE_PHRASES = [
    "the password is fortepiano",
    "my credit card number is one two three four five six",
    "the safe combination is forty seven sixteen",
    "transfer two thousand euros to account number ninety",
    "the access code for the vault is alpha tango bravo",
]


OUT_DIR = Path(__file__).resolve().parent / "hidden_phrases"
SPEECHBRAIN_CACHE = Path(
    os.environ.get(
        "SAFECOMMUTE_SPEECHBRAIN_CACHE",
        str(Path.home() / ".cache" / "safecommute-ai" / "speechbrain"),
    )
)
TACOTRON_SOURCE = "speechbrain/tts-tacotron2-ljspeech"
TACOTRON_SAVEDIR = str(SPEECHBRAIN_CACHE / "tts-tacotron2-ljspeech")
LJSPEECH_VOCODER_SOURCE = "speechbrain/tts-hifigan-ljspeech"
LJSPEECH_VOCODER_SAVEDIR = str(SPEECHBRAIN_CACHE / "tts-hifigan-ljspeech")
LJ_SAMPLE_RATE = 22050


def _load_models():
    from speechbrain.inference.TTS import Tacotron2
    from speechbrain.inference.vocoders import HIFIGAN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tacotron2 = Tacotron2.from_hparams(
        source=TACOTRON_SOURCE, savedir=TACOTRON_SAVEDIR,
        run_opts={"device": device},
    )
    vocoder = HIFIGAN.from_hparams(
        source=LJSPEECH_VOCODER_SOURCE, savedir=LJSPEECH_VOCODER_SAVEDIR,
        run_opts={"device": device},
    )
    return tacotron2, vocoder, device


def _synthesize_phrase(tacotron2, vocoder, device, text: str) -> np.ndarray:
    """LJSpeech 22.05 kHz mono. Returns float32 audio."""
    mel, _, _ = tacotron2.encode_text(text)
    with torch.no_grad():
        wav = vocoder.decode_batch(mel).squeeze().cpu().numpy().astype(np.float32)
    return wav


def _fit_to_target(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample to 16 kHz, then pad or center-crop to exactly 3 s."""
    if src_sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=SAMPLE_RATE)
    if len(audio) >= TARGET_LENGTH:
        start = (len(audio) - TARGET_LENGTH) // 2
        return audio[start:start + TARGET_LENGTH].astype(np.float32)
    pad = TARGET_LENGTH - len(audio)
    return np.pad(audio, (pad // 2, pad - pad // 2)).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if len(m.get("clips", [])) == len(PROBE_PHRASES) and all(
            (OUT_DIR / f"{c['clip_id']}.wav").exists() for c in m["clips"]
        ):
            print(f"already have {len(m['clips'])} probe clips at {OUT_DIR}; skipping")
            return

    tacotron2, vocoder, device = _load_models()
    print(f"synthesising {len(PROBE_PHRASES)} probe clips on device={device}")

    clips_meta = []
    for i, phrase in enumerate(PROBE_PHRASES):
        clip_id = f"probe-{i:02d}"
        audio = _synthesize_phrase(tacotron2, vocoder, device, phrase)
        audio = _fit_to_target(audio, src_sr=LJ_SAMPLE_RATE)
        sf.write(str(OUT_DIR / f"{clip_id}.wav"), audio, SAMPLE_RATE)
        clips_meta.append({
            "clip_id": clip_id,
            "speaker_id": "ljspeech_synth",
            "chapter_id": "synthetic",
            "transcript": phrase.upper(),
        })
        print(f"  {clip_id}: \"{phrase}\"  ({len(audio)} samples)")

    manifest = {
        "corpus": "hidden_phrases",
        "source": f"{TACOTRON_SOURCE} + {LJSPEECH_VOCODER_SOURCE}, resampled to {SAMPLE_RATE} Hz",
        "sample_rate": SAMPLE_RATE,
        "clip_duration_sec": 3.0,
        "target_length_samples": TARGET_LENGTH,
        "n_clips": len(clips_meta),
        "clips": clips_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(clips_meta)} probe clips + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
