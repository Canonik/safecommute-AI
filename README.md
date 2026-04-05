# SafeCommute AI

Privacy-first edge audio classifier for detecting escalation (aggressive shouting, distress screams) in public spaces. Built by students at Bocconi University.

**Raw audio is never recorded, stored, or transmitted.** Only non-reconstructible mel spectrograms are processed on-device. GDPR-compliant by architecture.

Deployable to any acoustic environment via fine-tuning. Base model trained on universal threat sounds + hard negatives. Per-deployment fine-tuning adapts the "safe" class to the target environment (metro, bar, bus) in minutes.

---

## Results (v2 — clean data, no acted speech, no leakage)

| Metric | Train | Val | Test | Gap |
|--------|-------|-----|------|-----|
| AUC-ROC | 0.928 | 0.819 | **0.856** | 0.072 |
| Accuracy | 0.746 | 0.660 | 0.675 | 0.071 |
| F1 | 0.752 | 0.660 | 0.684 | 0.068 |

**3-seed evaluation**: AUC = 0.844 +/- 0.019, Accuracy = 0.639 +/- 0.038, F1 = 0.644 +/- 0.043.

7 MB float32, 5 MB INT8, ~7 ms CPU inference. Deployment test: 100% threat detection rate.

### Per-Source Accuracy (test set)

| Source | Accuracy | Type |
|--------|----------|------|
| YouTube screams | 87.6% | Real-world |
| Hard negatives (HNS) | 88.0% | Environmental |
| Violence dataset | 72.3% | Real-world |
| AudioSet (all categories) | 65.0% | AudioSet |
| YouTube metro | 62.9% | Real-world |
| UrbanSound8K (bg) | 61.5% | Environmental |
| ESC-50 | 48.1% | Environmental |

### SOTA Comparison

| Model | Params | Size | Latency | AUC |
|-------|--------|------|---------|-----|
| **SafeCommute (ours)** | **1.83M** | **7MB** | **12ms** | **0.856** |
| PANNs CNN14 | 81.8M | 320MB | 250ms | 0.624 |
| AST Transformer | 86.6M | 330MB | 965ms | 0.615 |
| Wav2Vec2 | 94.4M | 360MB | 192ms | 0.523 |

Our model outperforms all SOTA baselines on this task while being 44x smaller and 21x faster.

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

### Training Recipe

- **Loss**: Focal loss (gamma=3) with dynamic class weights + label smoothing (0.1)
- **Augmentation**: SpecAugment (freq/time masking) per-sample in DataLoader + batch GPU ops (Gaussian noise, time shift, frequency dropout)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4), cosine annealing warm restarts
- **Regularization**: Dropout 0.3, gradient clipping (max_norm=1.0), early stopping (patience=6)
- **Data**: Clean spectrograms only (no augmentation at prep time)

---

## Data Strategy (v2)

Training data is organized in three layers:

**Layer 1 — Universal threat sounds (unsafe):** AudioSet (Screaming, Shout, Yell, Gunshot, Explosion, Glass breaking) + YouTube real screams + violence dataset

**Layer 2 — Universal hard negatives (safe):** AudioSet (Laughter, Crowd, Speech, Music, Applause, Cheering, Singing) + ESC-50 + UrbanSound8K

**Layer 3 — Deployment-specific ambient (safe, fine-tuning only):** Recorded in-situ audio per deployment vertical (metro rides, bar ambience, bus rides, etc.)

| Split | Safe | Unsafe | Total |
|-------|------|--------|-------|
| Train | 19,328 | 9,444 | 28,772 |
| Val | 3,587 | 2,026 | 5,613 |
| Test | 4,541 | 2,048 | 6,589 |

Acted speech datasets (CREMA-D, SAVEE, TESS, RAVDESS) were dropped — acted emotions sound fundamentally different from real escalation (35-52% accuracy proved they damage the model).

See `research/data_sources.md` for full citations and details.

---

## Deployment Personalization

The base model detects threats universally. To adapt for a specific environment:

```bash
# Fine-tune for metro (freeze CNN, train GRU+FC only)
PYTHONPATH=. python safecommute/pipeline/finetune.py \
    --environment metro --ambient-dir raw_data/youtube_metro --freeze-cnn

# Test deployment readiness
PYTHONPATH=. python safecommute/pipeline/test_deployment.py \
    --model models/metro_model.pth
```

This adapts the "safe" class boundary to the target acoustic environment in minutes.

---

## Quick Start

```bash
# Setup
sudo pacman -S portaudio ffmpeg python-pip    # Arch/CachyOS
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# Download and prepare data
PYTHONPATH=. python safecommute/pipeline/download_datasets.py    # ESC-50
PYTHONPATH=. python safecommute/pipeline/download_audioset.py    # AudioSet
PYTHONPATH=. python safecommute/pipeline/data_pipeline.py        # Prepare features
PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py # YouTube
PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py # Violence
PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py      # Verify

# Train
PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 3.0

# Analyze
PYTHONPATH=. python safecommute/pipeline/analyze.py

# Live demo
PYTHONPATH=. python safecommute/pipeline/inference.py
```

---

## Research Summary

### Ablation Study (v2)

| Variant | AUC | Params | Delta |
|---------|-----|--------|-------|
| Full model | 0.827 | 1.83M | baseline |
| No SE attention | 0.849 | 1.82M | +0.022 |
| No GRU | 0.843 | 1.16M | +0.016 |
| No multi-scale pool | 0.855 | 1.83M | +0.028 |
| Half channels | 0.836 | 458K | +0.009 |

### LOSO Key Finding

When held out entirely from training: laughter (6% accuracy), crowd (11%), speech (20%) are near-completely misclassified. These are essential hard negatives. Threat sounds generalize well across sources (gunshot 90%, screaming 81%).

See `research/experiment_log.md` for full results.

---

## Team

- **Alessandro Canonico** -- Project Lead & AI Strategist
- **Fabiola Martignetti** -- Behavioral Data & ML Specialist
- **Robbie Urquhart** -- Machine Learning & Edge Engineer

## License

See [LICENSE](LICENSE) for details.
