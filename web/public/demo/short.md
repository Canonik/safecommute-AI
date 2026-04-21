# SafeCommute AI — Demo model

Binary audio classifier for public-safety escalation. Edge-only, privacy-preserving, ~3–30 ms CPU inference (hardware-dependent; see latency table below).

**This bundle contains:**

- `safecommute_v2_int8.onnx` — static-INT8 ONNX model (1.83 M params, 3.72 MB on disk)
- `feature_stats.json` — `{mean, std}` normalization stats used by `infer.py`
- `infer.py` — standalone inference runner (no PyTorch required)
- `short.md` — this file
- `README-bundle.md` — install + run instructions

## What it does

Given a 3-second audio window at 16 kHz, the model returns a scalar in [0, 1] representing the probability the window contains an unsafe / escalating event (shouting, aggression, screaming) versus ambient or normal speech.

It is a **demo** model. On raw, ambient public-space audio it produces ~72% false-positive on speech. It reaches deployment quality only after per-site fine-tuning on ~30 minutes of ambient recordings from the target environment.

## Architecture

```
raw audio (16 kHz mono, 3 s, 48 000 samples)
        │
        ▼
mel spectrogram → PCEN normalization → z-score
        │                    (mean/std from feature_stats.json)
        ▼
(1, 1, 64, 188)  ─────────  input tensor
        │
        ▼
CNN6 encoder   3 Conv blocks (1 → 64 → 128 → 256)
               + Squeeze-and-Excitation on frequency
               + BatchNorm + ReLU + MaxPool
        │
        ▼  (B, 256, T')
GRU   256 → 128 hidden units (temporal escalation)
        │
        ▼
multi-scale pool   concat(last_hidden, mean, max) → (B, 384)
        │
        ▼
FC   384 → 2 logits (safe / unsafe)
```

- Parameters: **1,829,444 (1.83 M)**
- File size: **7.00 MB** float32 / **3.72 MB** INT8 ONNX (shipped in this bundle)
- Inference latency (hardware-disclosed, model-only):
  - ~12 ms on Ryzen 5, 1 thread — *historical* (original measurement setup; hardware not on hand for reproduction)
  - 4.0 ms FP32 ONNX / **2.8 ms INT8 ONNX** on Ryzen 7 7435HS, 8 threads (measured 2026-04-21)
  - 26.9 ms FP32 ONNX on Ryzen 7 7435HS, 1 thread (single-thread comparable)
- End-to-end (including librosa PCEN preprocessing): ~20–50 ms on Ryzen 7; preprocessing dominates. Raspberry Pi numbers pending.
- No GPU required

## Quickstart (Python)

```bash
pip install torch librosa numpy
```

```python
import torch, torch.nn as nn
import librosa, numpy as np

# Assumes you have cloned https://github.com/Canonik/safecommute-AI
# which contains safecommute/model.py (SafeCommuteCNN) and safecommute/features.py
from safecommute.model import SafeCommuteCNN
from safecommute.features import extract_features  # PCEN + z-score

model = SafeCommuteCNN()
model.load_state_dict(torch.load("safecommute_v2.pth", map_location="cpu", weights_only=True))
model.eval()

audio, sr = librosa.load("clip.wav", sr=16000, mono=True, duration=3.0)
if len(audio) < 48000:
    audio = np.pad(audio, (0, 48000 - len(audio)))
x = extract_features(audio[:48000])          # (1, 64, 188)
x = torch.from_numpy(x).unsqueeze(0).float() # (1, 1, 64, 188)

with torch.no_grad():
    p_unsafe = torch.softmax(model(x), dim=-1)[0, 1].item()

print(f"p(unsafe) = {p_unsafe:.3f}")
```

## Input contract (strict)

| Field       | Value                                       |
|-------------|---------------------------------------------|
| Sample rate | 16 000 Hz                                   |
| Channels    | mono                                        |
| Window      | 3.0 s (48 000 samples)                      |
| Features    | 64-bin mel + PCEN + per-feature z-score     |
| Tensor      | `(batch, 1, 64, 188)` float32               |

Running on other sample rates or window lengths will silently degrade accuracy.

## Privacy guarantee

The input representation is a **PCEN mel spectrogram**. This transform is non-invertible — phase is discarded and amplitude is log-compressed with adaptive gain. Speech cannot be reconstructed from the tensor the model sees. Nothing is sent to the cloud.

## When it fails

Out of the box on raw ambient audio (measured 2026-04-21, `tests/reports/verify_performance_claims.json`):

- **71.7% false-positive on speech** (`as_speech` subset, n=378)
- **82.5% false-positive on laughter** (`as_laughter`, n=269) — the docs historically highlighted speech; laughter is worse.
- Poor robustness to HVAC / wheel squeal / platform noise
- No site-specific calibration

**After fine-tuning**, on the one real site we have measured (metro, 58 fine-tune clips + 34 truly-held-out quarantine clips), the default `--freeze-cnn` recipe with the `low_fpr` threshold produces **38.2% FP / 89.5% threat recall** on the held-out set — threat recall matches the intent, FP does not. A tweak sweep (see `tests/reports/tweak_finetune.json`) is in the repo; results propagate into the paper's Limitations section. **Do not cite a specific post-fine-tune FP number without checking this JSON.** One site's numbers are not n=3; additional site recordings are future work.

## Fine-tuning

The paid dashboard lets you:

1. Upload 30+ minutes of ambient recordings from your site.
2. Trigger a fine-tune job (CNN frozen, only the GRU + FC head are adapted).
3. Download the calibrated `{site}_model.pth` and matching `{site}_thresholds.json`.

Command-line equivalent (requires the full repo):

```bash
PYTHONPATH=. python safecommute/pipeline/finetune.py \
  --environment metro \
  --ambient-dir raw_data/metro/ \
  --freeze-cnn
```

## Links

- Repository: <https://github.com/Canonik/safecommute-AI>
- Full docs: `README.md`, `DEPLOY.md` in the repo
- Fine-tuning dashboard: open the landing page → "Open dashboard"

## License & attribution

Model weights and code released under the MIT license. If you deploy this in production — for anything that affects a real person — you must fine-tune it on your site. This is not negotiable.
