# SafeCommute AI

Privacy-first edge audio classifier for detecting escalation (aggressive shouting, distress screams) in public transport. Built by students at Bocconi University.

**Raw audio is never recorded, stored, or transmitted.** Only non-reconstructible mel spectrograms are processed on-device. GDPR-compliant by architecture.

---

## Results

| Metric | Train | Val | Test | Gap |
|--------|-------|-----|------|-----|
| AUC-ROC | 0.978 | 0.959 | **0.950** | 0.028 |
| Accuracy | 0.771 | 0.768 | 0.745 | 0.026 |
| F1 | 0.785 | 0.782 | 0.760 | 0.025 |

**No overfitting** (train-test AUC gap = 0.028). Noise-robust (AUC >0.94 at Gaussian noise sigma=0.2). 7 MB float32, ~7 ms CPU inference.

With optimized threshold (Youden's J = 0.622): **Accuracy = 86.0%, F1 = 0.865, Sensitivity = 89.2%**.

### Per-Source Accuracy

| Source | Accuracy | Type |
|--------|----------|------|
| YouTube screams | 97.3% | Real-world |
| TESS | 98.8% | Acted speech |
| Hard negatives (HNS) | 97.1% | Environmental |
| YouTube metro | 90.3% | Real-world |
| ESC-50 | 71.9% | Environmental |
| Violence dataset | 71.0% | Real-world |
| RAVDESS | 52.1% | Acted speech |
| CREMA-D | 45.2% | Acted speech |

The model excels on real-world audio but struggles with acted speech — acted anger sounds fundamentally different from real escalation.

---

## Model Architecture

**CNN6 + Squeeze-and-Excitation + GRU + Multi-Scale Pooling** (1,829,444 parameters)

```
Input: (B, 1, 64, 188) log-Mel spectrogram
       3-second audio @ 16kHz, n_mels=64, hop_length=256

ConvBlock1: 2x Conv2d(1->64, 3x3) + BN + ReLU + SE(64) + AvgPool(2x2)
            Output: (B, 64, 32, 94)

ConvBlock2: 2x Conv2d(64->128, 3x3) + BN + ReLU + SE(128) + AvgPool(2x2)
            Output: (B, 128, 16, 47)

ConvBlock3: 2x Conv2d(128->256, 3x3) + BN + ReLU + SE(256) + AvgPool(2x2)
            Output: (B, 256, 8, 23)

Reshape:    (B, 256, 8, 23) -> (B, 23, 2048)
FreqReduce: Linear(2048->256) + ReLU -> (B, 23, 256)

GRU:        GRU(256->128, 1 layer, batch_first)
            Output: (B, 23, 128), hidden: (1, B, 128)

Multi-Scale Pooling:
  last_hidden = GRU hidden state -> (B, 128)
  mean_pool   = mean over time   -> (B, 128)
  max_pool    = max over time    -> (B, 128)
  concat -> (B, 384)

Classifier: Dropout(0.3) -> Linear(384->2)
```

### Why This Architecture

- **CNN6-style blocks** (from PANNs, Kong et al. 2020): double-conv blocks designed for mel spectrograms, not repurposed ImageNet architectures
- **SE attention**: learns which frequency bands matter (e.g., 2-4 kHz for screams), negligible parameter cost
- **GRU**: captures temporal escalation patterns (rising pitch, increasing intensity) that frame-level CNNs miss
- **Multi-scale pooling**: combines endpoint (last hidden) with aggregate statistics (mean/max), capturing both how the audio ends and its overall character

### Training Recipe

- **Loss**: Focal loss (gamma=3) with dynamic class weights + label smoothing (0.1)
- **Augmentation**: SpecAugment (freq/time masking), Gaussian noise, circular time shift, mixup (alpha=0.3)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4), cosine annealing warm restarts (T0=5, Tmult=2)
- **Regularization**: Dropout 0.3, gradient clipping (max_norm=1.0), early stopping (patience=6)

---

## Data

16,037 training / 3,439 validation / 3,472 test samples from 9 sources.

| Source | Train | Val | Test | Type | Safe/Unsafe Mapping |
|--------|-------|-----|------|------|---------------------|
| CREMA-D | 4,301 | 884 | 986 | Acted speech (91 actors) | ANG,FEA -> unsafe; HAP,NEU,SAD -> safe |
| YouTube | 3,300 | 721 | 691 | Real metro + screams | Metro ambient -> safe; screams -> unsafe |
| TESS | 1,097 | 256 | 247 | Acted speech (2 speakers) | angry,fear -> unsafe; neutral,happy -> safe |
| HNS | 2,385 | 526 | 518 | Environmental sounds | All safe (loud but non-threatening) |
| UrbanSound8K | 2,125 | 445 | 430 | Urban backgrounds | All safe |
| Violence | 1,407 | 305 | 300 | Violence detection dataset | label 0 -> safe; label 1 -> unsafe |
| RAVDESS | 594 | 126 | 144 | Acted speech (24 actors) | emotion 05,06 -> unsafe; 01,02,03 -> safe |
| ESC-50 | 566 | 120 | 114 | Environmental sounds | All safe |
| SAVEE | 262 | 56 | 42 | Acted speech (4 speakers) | angry,fear -> unsafe; rest -> safe |

**Class balance**: 74% safe / 26% unsafe, handled by focal loss with inverse-frequency class weights.

### Feature Format

All audio: 16kHz mono, 3-second clips (pad or random crop). Stored as pre-computed `.pt` tensors:
- Log-mel spectrogram: `librosa.feature.melspectrogram()` with ref=1.0 to preserve loudness
- Shape: `(1, 64, 188)` — 1 channel, 64 mel bins, 188 time frames
- Normalized by global mean/std (stored in `feature_stats.json`)

---

## Repository Structure

```
safecommute/                         # Everything lives here
  model.py                           # SafeCommuteCNN architecture
  constants.py                       # Shared constants (sample rate, mel bins, etc.)
  features.py                        # Mel spectrogram extraction
  dataset.py                         # TensorAudioDataset (loads .pt tensors)
  utils.py                           # Seed everything for reproducibility
  export.py                          # INT8, ONNX, TorchScript export
  distill.py                         # PANNs knowledge distillation
  domain_adversarial.py              # Domain-adversarial training

  pipeline/                          # Data preparation, training, analysis
    download_datasets.py             # Download all 9 datasets
    data_pipeline.py                 # Prepare mel spectrograms from raw audio
    prepare_youtube_data.py          # YouTube audio extraction (yt-dlp)
    prepare_violence_data.py         # Violence dataset preparation
    validate_youtube_data.py         # Automated data quality validation
    train.py                         # Training (focal loss, augmentation, mixup)
    analyze.py                       # Full analysis: ROC, calibration, per-source
    inference.py                     # Live microphone inference

  benchmark/                         # SOTA comparison benchmark suite
    run_benchmark.py                 # Run all benchmarks
    metrics.py                       # Evaluation metrics
    profiler.py                      # Latency/memory profiling
    models/                          # Wrappers for PANNs, AST, Wav2Vec2, etc.

research/                            # Research experiments (sandboxed)
  README.md                          # Results overview
  NEXT_STEPS.md                      # Publication roadmap
  literature_review.md               # 20-paper survey
  experiment_log.md                  # All experiment results
  robustness_report.md               # Overfitting/noise/calibration analysis
  dataset_audit_report.md            # Data quality audit
  experiments/                       # 15 experiment scripts

demo.py                              # Quick 15-second live demo
```

---

## Quick Start

```bash
# Setup
sudo pacman -S portaudio ffmpeg python-pip    # Arch/CachyOS
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# Download and prepare data
PYTHONPATH=. python safecommute/pipeline/download_datasets.py
PYTHONPATH=. python safecommute/pipeline/data_pipeline.py
PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py
PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py
PYTHONPATH=. python safecommute/pipeline/validate_youtube_data.py

# Train
PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 3.0

# Analyze
PYTHONPATH=. python safecommute/pipeline/analyze.py

# Live demo
PYTHONPATH=. python demo.py
```

---

## Research Summary

12 experiments completed comparing our architecture against alternative techniques and SOTA baselines. Full details in `research/README.md`.

| Rank | Experiment | AUC | Accuracy | Params | Key Finding |
|------|-----------|-----|----------|--------|-------------|
| 1 | Knowledge Distillation | **0.954** | **76.4%** | 1.83M | Best AUC via self-distillation |
| 2 | Baseline (production) | 0.950 | 74.5% | 1.83M | Current model |
| 3 | Depthwise Separable | 0.944 | 75.0% | 819K | 55% fewer params |
| 4 | SOTA: Audio ResNet | 0.929 | 76.1% | 816K | Higher acc, lower AUC |
| 5 | SOTA: Simple CNN | 0.881 | 64.5% | 93K | Proves architecture matters |

**Our CNN6+SE+GRU outperforms ResNet, Transformer, BiLSTM, and Simple CNN** on this task.

### Validation (in progress)

| Test | Result | Status |
|------|--------|--------|
| 5-fold CV (source-aware) | AUC = 0.944 +/- 0.006 | **Done** |
| Threshold optimization | Acc 74.5% -> 86.0% at t=0.622 | **Done** |
| Ablation study | GRU most critical (-0.024 AUC without) | **Done** |
| Leave-one-source-out | Mean AUC=0.767, YouTube=0.803 | **Done** |

### Next Steps

1. **Ablation study**: prove each component (SE, GRU, multi-scale pooling) contributes
2. **LOSO evaluation**: train on 8 sources, test on held-out 1
3. **Domain adaptation**: MMD loss or DANN for acted-to-real gap
4. **Real-world validation**: test on actual public transport audio

See `research/NEXT_STEPS.md` for the full roadmap.

---

## Team

- **Alessandro Canonico** -- Project Lead & AI Strategist
- **Fabiola Martignetti** -- Behavioral Data & ML Specialist
- **Robbie Urquhart** -- Machine Learning & Edge Engineer

## License

See [LICENSE](LICENSE) for details.
