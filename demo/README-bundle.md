# SafeCommute AI — Demo Bundle

Standalone CPU inference using the base INT8 ONNX model. No PyTorch required.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install numpy scipy librosa onnxruntime
```

## Run

```bash
python infer.py path/to/sample.wav
```

Output:

```
file           : sample.wav
model          : safecommute_v2_int8.onnx
P(unsafe)      : 0.184   (threshold 0.50)
prediction     : safe
latency (ms)   : load 40.2  preprocess 18.7  model 3.2  total 62.1
```

## Contents

| File | Size | Purpose |
|---|---|---|
| `safecommute_v2_int8.onnx` | ~3.7 MB | Static INT8-quantized model, CPU-only |
| `infer.py` | ~5 KB | Standalone runner (argparse + librosa PCEN + onnxruntime) |
| `feature_stats.json` | <100 B | `{mean, std}` normalization stats baked into training |
| `short.md` | ~5 KB | Architecture / privacy / fine-tune notes |
| `README-bundle.md` | this file | Install + run instructions |

## Deployment notes

- **Base model vs fine-tuned**: this zip ships the *base* model trained on
  universal threat sounds (AudioSet screams, UrbanSound8K, ESC-50) plus
  universal hard negatives (speech, laughter, crowd). Its FP rate on
  everyday speech is ~72% *by design* — the speech/shout spectral overlap
  is fundamental. **Per-site fine-tuning with 30+ min of your environment's
  ambient audio is required** for deployable accuracy. See `short.md` §4.
- **Threshold**: default 0.5 is a coarse starting point. If you fine-tune
  for your site, use the `low_fpr` value from the generated
  `thresholds.json` (expect 0.65–0.80 range).
- **Latency**: model-only inference is 2–4 ms on a modern x86 CPU at 8
  threads; preprocessing (librosa mel + PCEN) adds 15–25 ms. Total
  end-to-end is dominated by preprocessing. On a Raspberry Pi 4/5 expect
  ~3–10× slower preprocessing — see the paper's hardware table (pending).

## Privacy boundary

`extract_pcen(y)` is **non-invertible** by construction (PCEN's adaptive
gain control destroys phase and per-band energy envelope). Once audio
crosses that function, the original waveform cannot be recovered. This is
the privacy guarantee, not a policy promise — the transform is physically
one-way.

What this script does NOT do on your behalf:
- Persist raw audio to disk (it's in RAM only for the duration of one call).
- Send anything to a network.
- Compute per-frame identifiers.

If you build a production deployment on top of this runner, keep the PCEN
boundary: feed 3-second windows of audio from a rolling RAM buffer, never
commit the raw waveform to storage.

## Verifying the artefact

```bash
python -c "import onnxruntime as ort; s = ort.InferenceSession('safecommute_v2_int8.onnx'); print(s.get_inputs()[0].shape)"
# [1, 1, 64, 188]
```

If this prints without a "missing external data" warning, the bundle is
self-contained. The upstream repo's `safecommute/export_quantized.py` is
the single source of this ONNX — its SHA256 is listed in
`tests/reports/artefacts.sha256` on the repo.
