# SafeCommute AI

## Privacy-First Audio AI for Escalation Detection in Public Transport

SafeCommute AI is an on-device machine learning system that detects acoustic signs of escalation (aggressive shouting, distress screams) in public transport. Built by students at Bocconi University.

**Privacy by design**: Raw audio is never recorded, stored, or transmitted. Only non-reconstructible Mel spectrograms are processed on-device. GDPR-compliant by architecture.

---

## Performance

### Model Metrics (held-out test set, 3,472 samples)

| Metric | Train | Val | Test | Gap |
|--------|-------|-----|------|-----|
| AUC-ROC | 0.978 | 0.959 | **0.950** | 0.028 |
| Accuracy | 0.771 | 0.768 | 0.745 | 0.026 |
| F1 | 0.785 | 0.782 | 0.760 | 0.025 |

**Overfitting assessment**: Train-test gap of 0.026 accuracy / 0.028 AUC — no significant overfitting. The model generalizes well.

### Comparison with SOTA Models (same test set)

| Model | Params | Size | Latency | AUC-ROC |
|---|---|---|---|---|
| **SafeCommute (ours)** | **1.83M** | **7 MB** | **7 ms** | **0.950** |
| SafeCommute INT8 | 1.15M | 5 MB | 7 ms | 0.950 |
| PANNs CNN14 | 81.8M | 320 MB | 250 ms | 0.660 |
| Energy Baseline | 0 | 0 | <1 ms | 0.503 |

**45x smaller** than PANNs CNN14, **36x faster**, **1.4x higher AUC** on this domain-specific task.

### Per-Source Accuracy

| Source | Accuracy | Type | Assessment |
|--------|----------|------|------------|
| YouTube screams | 97.3% | Real-world | Strong |
| TESS | 98.8% | Acted speech | Strong |
| Hard negatives | 97.1% | Environmental | Strong |
| YouTube metro | 90.3% | Real-world | Strong |
| UrbanSound8K | 86.0% | Environmental | Moderate |
| Violence dataset | 71.0% | Real-world | Moderate |
| RAVDESS | 52.1% | Acted speech | Weak |
| CREMA-D | 45.2% | Acted speech | Weak |

**Key insight**: The model excels on real-world audio (97% on YouTube screams) but struggles with acted speech from controlled datasets. This is because acted anger sounds fundamentally different from real escalation — the model correctly learned real-world patterns rather than memorizing actor performances.

### Known Limitations

- **Acted speech weakness**: CREMA-D/RAVDESS actors perform "anger" differently from real aggression
- **No Italian-specific model**: Emozionalmente (6,902 Italian clips) is available but degrades overall performance when mixed
- **No field validation**: Never tested on actual metro platform hardware
- **Single microphone**: No spatial awareness or distance estimation
- **3-second window**: Brief escalations may be missed

---

## Architecture

**CNN6 + SE + GRU + Multi-Scale Pooling** (1,829,444 parameters)

```
Input: (B, 1, 64, 188) log-Mel spectrogram — 3s @ 16kHz, hop=256
  → 3× ConvBlock (2×Conv2d + BN + ReLU + SE attention + AvgPool)
  → Linear freq projection (2048 → 256)
  → GRU (256 → 128, 1 layer)
  → Multi-scale pooling: concat(last_hidden, mean_pool, max_pool) → 384
  → Dropout(0.3) → FC(384 → 2)
```

**Training**: Focal loss (γ=3), cosine annealing warm restarts, strong spectrogram augmentation (frequency/time masking, Gaussian noise, time shift), mixup (α=0.3).

---

## Training Data

16,037 samples from 8 sources, validated for quality:

| Source | Clips | Type | Safe/Unsafe mapping |
|--------|-------|------|---------------------|
| CREMA-D | 7,442 | Acted speech (91 actors) | ANG,FEA → unsafe; HAP,NEU,SAD → safe |
| TESS | 2,800 | Acted speech (2 speakers) | angry,fear → unsafe; neutral,happy → safe |
| RAVDESS | 1,440 | Acted speech (24 actors) | emotion 05,06 → unsafe; 01,02,03 → safe |
| SAVEE | 480 | Acted speech (4 speakers) | angry,fear → unsafe; happy,neutral,sad → safe |
| UrbanSound8K | ~6,400 | Urban sounds | Backgrounds + hard negatives → safe |
| ESC-50 | ~800 | Environmental | Ambient → safe; loud-but-safe → safe |
| YouTube | ~115 | Real metro + screams | Metro → safe; screams → unsafe |
| Violence | 2,000 | Violence detection | label 0 → safe; label 1 → unsafe |

**Data cleaning**: YouTube audio validated with automated quality checks (removed 38 files: music, news broadcasts, too-energetic ambient). All acted speech datasets use published emotion labels.

---

## Project Structure

```
safecommute/                  # Shared Python package
  model.py                    # Model architecture (single source of truth)
  constants.py                # All shared constants
  features.py                 # Mel spectrogram extraction
  dataset.py                  # PyTorch dataset
  utils.py                    # Reproducibility
  export.py                   # INT8, ONNX, TorchScript export
  distill.py                  # Knowledge distillation (PANNs teacher)
  domain_adversarial.py       # Domain-adversarial training

v_3/                          # Pipeline scripts
  data_pipeline_3.0.py        # Dataset preparation
  train_model_3.0.py          # Standard training
  train_experimental.py       # Ablation study training
  train_domain_adversarial.py # Domain-adversarial training
  mvp_inference_3.0.py        # Live microphone inference
  calibrate_thresholds_3.0.py # Threshold calibration
  mine_hard_negatives_3.0.py  # False positive mining
  download_datasets.py        # Dataset downloader
  prepare_youtube_data.py     # YouTube audio processor
  prepare_violence_data.py    # Violence dataset processor
  validate_youtube_data.py    # Data quality validation
  comprehensive_analysis.py   # Full model analysis + plots
  benchmark/                  # SOTA comparison suite

demo.py                       # Quick live demo (15s recording)
```

---

## Quick Start

```bash
# Setup (Arch/CachyOS)
sudo pacman -S portaudio ffmpeg python-pip
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# Download and prepare data
PYTHONPATH=. python v_3/download_datasets.py
PYTHONPATH=. python v_3/data_pipeline_3.0.py
PYTHONPATH=. python v_3/prepare_youtube_data.py
PYTHONPATH=. python v_3/prepare_violence_data.py
PYTHONPATH=. python v_3/validate_youtube_data.py

# Train
PYTHONPATH=. python v_3/train_experimental.py --focal --cosine --strong-aug --gamma 3.0

# Analyze (generates plots + report in analysis/)
PYTHONPATH=. python v_3/comprehensive_analysis.py

# Benchmark against SOTA
PYTHONPATH=. python v_3/benchmark/run_benchmark.py

# Live demo
PYTHONPATH=. python demo.py
```

---

## Team

- **Alessandro Canonico** — Project Lead & AI Strategist
- **Fabiola Martignetti** — Behavioral Data & ML Specialist
- **Robbie Urquhart** — Machine Learning & Edge Engineer

---

## License

See [LICENSE](LICENSE) for details.
