# SafeCommute AI

## Privacy-First Audio AI for Escalation Detection in Public Transport

SafeCommute AI is an on-device machine learning system that detects early acoustic signs of escalation (aggressive shouting, distress screams) in crowded public transport environments.

Built by students at Bocconi University, the system runs entirely on-device — raw audio is never recorded, stored, or transmitted. Only non-reconstructible Mel spectrograms are processed, making it GDPR-compliant by design.

---

## Benchmark Results

Evaluated on 4,413 held-out test samples against 5 SOTA audio models:

| Model | Params | Size | Latency | AUC-ROC |
|---|---|---|---|---|
| **SafeCommute (ours)** | **1.83M** | **7 MB** | **~9 ms** | **0.971** |
| SafeCommute INT8 | 1.15M | 5 MB | ~9 ms | 0.971 |
| AST (Transformer) | 86.6M | 330 MB | 442 ms | 0.678 |
| PANNs CNN14 | 81.8M | 320 MB | 250 ms | 0.658 |
| Whisper-tiny | 37.8M | 144 MB | 72 ms | 0.567 |
| Wav2Vec2 | 94.4M | 360 MB | 92 ms | 0.472 |

**Key advantages**: 45x smaller than PANNs, 50x faster than AST, 1.4x higher AUC than any general-purpose SOTA on this domain-specific task.

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

**Training**: Focal loss (γ=2), cosine annealing warm restarts, strong spectrogram augmentation (noise injection, time shift, frequency/time masking), mixup augmentation.

**Training data**: 18,597 samples from 8 sources:

| Source | Clips | Type |
|--------|-------|------|
| CREMA-D | 7,442 | Acted emotional speech (91 actors, 6 emotions) |
| TESS | 2,800 | Acted emotional speech (2 speakers, 7 emotions) |
| RAVDESS | 1,440 | Acted emotional speech (24 actors, 8 emotions) |
| SAVEE | 480 | Acted emotional speech (4 speakers, 7 emotions) |
| UrbanSound8K | ~6,400 | Urban environmental sounds + hard negatives |
| ESC-50 | ~800 | Environmental sound classification |
| YouTube | ~130 clips | Real metro ambient + real screams/confrontations |
| Violence Dataset | 2,000 | Real-world violence vs non-violence audio |

---

## Project Structure

```
safecommute/              # Shared Python package (single source of truth)
  model.py                # CNN6+SE+GRU model definition
  constants.py            # All shared constants
  features.py             # Mel spectrogram feature extraction
  dataset.py              # PyTorch dataset class
  utils.py                # Reproducibility seeds
  distill.py              # Knowledge distillation (PANNs CNN14 teacher)
  export.py               # INT8 quantization, ONNX, TorchScript
  domain_adversarial.py   # Gradient reversal + domain-adversarial model

v_3/                      # Active pipeline scripts
  data_pipeline_3.0.py    # Dataset download + feature preparation
  train_model_3.0.py      # Standard training (with optional --distill)
  train_experimental.py   # Ablation study (focal, cosine, augmentation)
  train_domain_adversarial.py  # Domain-adversarial training
  mvp_inference_3.0.py    # Live microphone inference
  calibrate_thresholds_3.0.py  # Threshold calibration
  mine_hard_negatives_3.0.py   # False positive mining
  download_datasets.py    # Dataset downloader
  prepare_youtube_data.py # YouTube audio processor
  prepare_violence_data.py # Violence dataset processor
  benchmark/              # Evaluation suite (5 SOTA models, 6 plots)

v_1/, v_2/                # Historical versions (reference only)
```

---

## Quick Start

```bash
# 1. Setup (Arch/CachyOS)
sudo pacman -S portaudio ffmpeg python-pip
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# 2. Download datasets
PYTHONPATH=. python v_3/download_datasets.py

# 3. Prepare features
PYTHONPATH=. python v_3/data_pipeline_3.0.py
PYTHONPATH=. python v_3/prepare_youtube_data.py
PYTHONPATH=. python v_3/prepare_violence_data.py

# 4. Train (best recipe)
PYTHONPATH=. python v_3/train_experimental.py --focal --cosine --strong-aug

# 5. Export optimized models
PYTHONPATH=. python -m safecommute.export

# 6. Benchmark against SOTA
PYTHONPATH=. python v_3/benchmark/run_benchmark.py

# 7. Live inference
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

## License

See [LICENSE](LICENSE) for details.
