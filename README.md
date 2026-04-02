# SafeCommute AI

## Privacy-First Audio AI for Early Detection of Escalation in Public Transport

SafeCommute AI is an on-device machine learning system that detects early acoustic signs of escalation (aggressive shouting, distress screams) in crowded public transport environments.

Built by students at Bocconi University, the system runs entirely on-device — raw audio is never recorded, stored, or transmitted. Only non-reconstructible Mel spectrograms are processed, making it GDPR-compliant by design.

---

## Benchmark Results

Evaluated on 2,481 held-out test samples against 5 SOTA audio models:

| Model | Params | Size | Latency | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| **SafeCommute (ours)** | **1.83M** | **7 MB** | **6.6 ms** | **86.9%** | **0.870** | **0.920** |
| SafeCommute INT8 | 1.15M | 5 MB | 6.7 ms | 86.9% | 0.870 | 0.920 |
| PANNs CNN14 | 81.8M | 320 MB | 250 ms | 57.9% | 0.425 | 0.653 |
| AST (Transformer) | 86.6M | 330 MB | 420 ms | 57.9% | 0.425 | 0.680 |
| Wav2Vec2 | 94.4M | 360 MB | 93 ms | 57.9% | 0.425 | 0.473 |
| Whisper-tiny | 37.8M | 144 MB | 75 ms | 42.1% | 0.249 | 0.568 |

**Key advantages**: 45x smaller than PANNs, 63x faster than AST, 2x higher accuracy than any general-purpose SOTA model on this domain-specific task.

---

## The Problem & Solution

**Challenge**: Security personnel in public transport hubs must monitor large, noisy spaces with limited real-time visibility.

**Barrier**: Traditional CCTV is reactive and raises GDPR/privacy concerns.

**SafeCommute Advantage**:
- Runs continuously on a 3-second sliding audio window
- Processes data directly on edge devices (Raspberry Pi, phones)
- Converts audio into non-reconstructible Mel spectrograms
- Never records, stores, or transmits raw audio
- Outputs only a compact alert: `Safe`, `Warning`, or `Alert`

---

## Team

- **Alessandro Canonico** — Project Lead & AI Strategist
- **Fabiola Martignetti** — Behavioral Data & ML Specialist
- **Robbie Urquhart** — Machine Learning & Edge Engineer

---

## Architecture

**Model**: CNN6 + Squeeze-and-Excitation + GRU + Multi-Scale Pooling (1.83M params)

```
Input (1, 64, 188) log-mel spectrogram
  → 3× ConvBlock (double-conv + BN + SE attention + avg-pool)
  → Linear freq projection (2048→256)
  → GRU (256→128, 1 layer)
  → Multi-scale pooling (last + mean + max → 384)
  → FC (384→2) → Safe / Unsafe
```

**Training data**: 11,330 samples from 5 datasets (RAVDESS, CREMA-D, TESS, SAVEE, UrbanSound8K) + ESC-50 environmental sounds, augmented with room reverb, SNR mixing, and SpecAugment.

---

## Project Structure

```
safecommute/              # Shared Python package
  model.py                # CNN6+SE+GRU model (single source of truth)
  constants.py            # All shared constants
  features.py             # Feature extraction (mel spectrograms)
  dataset.py              # PyTorch dataset class
  utils.py                # Reproducibility seeds
  distill.py              # Knowledge distillation from PANNs CNN14
  export.py               # INT8 quantization, ONNX, TorchScript

v_3/                      # Active pipeline scripts
  data_pipeline_3.0.py    # Dataset download + feature preparation
  train_model_3.0.py      # Training (with optional --distill flag)
  mvp_inference_3.0.py    # Live microphone inference
  calibrate_thresholds_3.0.py
  mine_hard_negatives_3.0.py
  download_datasets.py    # Standalone dataset downloader
  benchmark/              # Evaluation suite
    run_benchmark.py      # Benchmarks against 5 SOTA models
    results/              # Tables, JSON, 6 visualization plots

v_1/, v_2/                # Historical versions (reference only)
```

---

## Quick Start

```bash
# 1. Setup
sudo pacman -S portaudio ffmpeg python-pip    # Arch/CachyOS
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# 2. Download datasets & prepare features
PYTHONPATH=. python v_3/download_datasets.py
PYTHONPATH=. python v_3/data_pipeline_3.0.py

# 3. Train
PYTHONPATH=. python v_3/train_model_3.0.py

# 4. Export optimized models
PYTHONPATH=. python -m safecommute.export

# 5. Benchmark against SOTA
PYTHONPATH=. python v_3/benchmark/run_benchmark.py

# 6. Live inference (requires microphone)
PYTHONPATH=. python v_3/mvp_inference_3.0.py
```

---

## Tech Stack

- **Language**: Python 3.10+
- **ML**: PyTorch, torchaudio
- **Signal Processing**: librosa, NumPy, SciPy, pyroomacoustics
- **Audio I/O**: PyAudio
- **Benchmarking**: PANNs, HuggingFace Transformers (AST, Wav2Vec2, Whisper)
- **Deployment**: INT8 quantization, TorchScript, ONNX

---

## Troubleshooting

- **Risk score stuck at ~0.47**: Microphone is muted or disconnected.
- **CREMA-D download fails**: Run `v_3/download_datasets.py` which uses HuggingFace Hub.
- **PYTHONPATH errors**: Always run from repo root with `PYTHONPATH=.`

---

## License

See [LICENSE](LICENSE) for details.
