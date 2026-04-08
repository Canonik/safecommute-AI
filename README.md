# SafeCommute AI

Privacy-first edge audio classifier for detecting escalation (aggressive shouting, distress screams) in public spaces. Built by students at Bocconi University.

**Raw audio is never recorded, stored, or transmitted.** Only non-reconstructible PCEN spectrograms are processed on-device. GDPR-compliant by architecture.

Deployable to any acoustic environment via fine-tuning. Base model trained on universal threat sounds + hard negatives. Per-deployment fine-tuning adapts the "safe" class to the target environment (metro, bar, bus) in minutes.

---

## Current Best Model (Cycle 6 — gamma=0.5 + noise injection)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.804 |
| Accuracy | 70.3% |
| F1 | 0.716 |
| Parameters | 1.83M |
| Size | 7MB float32, 5MB INT8 |
| CPU Latency | ~12ms |

### Per-Source Accuracy

| Source | Accuracy | Type |
|--------|----------|------|
| as_yell | 90.6% | Threat |
| as_screaming | 79.1% | Threat |
| yt_scream | 78.2% | Threat |
| as_shout | 64.7% | Threat |
| yt_metro | 64.9% | Safe ambient |
| as_crowd | 42.1% | Safe hard negative |
| as_speech | 28.3% | Safe hard negative |
| as_laughter | 17.5% | Safe hard negative |

### Known Limitation

Speech, laughter, and crowd noise are still misclassified as threats too often. The model lacks diverse speech training data. Next step: integrate LibriSpeech, CommonVoice, and VoxCeleb as safe class to fix the speech false positive problem.

---

## Model Architecture

**CNN6 + Squeeze-and-Excitation + GRU + Multi-Scale Pooling** (1,829,444 parameters)

```
Input: (B, 1, 64, 188) PCEN spectrogram
       3-second audio @ 16kHz, n_mels=64, hop_length=256

ConvBlock1: 2x Conv2d(1->64, 3x3) + BN + ReLU + SE(64) + AvgPool(2x2)
ConvBlock2: 2x Conv2d(64->128, 3x3) + BN + ReLU + SE(128) + AvgPool(2x2)
ConvBlock3: 2x Conv2d(128->256, 3x3) + BN + ReLU + SE(256) + AvgPool(2x2)
FreqReduce: Linear(2048->256) + ReLU
GRU:        GRU(256->128, 1 layer, batch_first)
Multi-Scale: concat(last_hidden, mean_pool, max_pool) -> (B, 384)
Classifier: Dropout(0.3) -> Linear(384->2)
```

### Training Recipe

- **Loss**: Focal loss (gamma=0.5) with dynamic class weights
- **Augmentation**: SpecAugment (freq/time masking) + batch GPU ops (noise, time shift, freq dropout) + environmental noise injection (metro ambient at 0-20dB SNR)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4), cosine annealing warm restarts
- **Regularization**: Dropout 0.3, gradient clipping (max_norm=1.0), early stopping (patience=6)

---

## Data Strategy (v2)

**Layer 1 — Universal threats (unsafe):** AudioSet (Screaming, Shout, Yell, Gunshot, Explosion, Glass breaking) + YouTube real screams + violence dataset

**Layer 2 — Hard negatives (safe):** AudioSet (Laughter, Crowd, Speech, Music, Applause, Cheering, Singing) + ESC-50 + UrbanSound8K

**Layer 3 — Deployment ambient (safe, fine-tuning only):** Recorded in-situ audio per deployment

| Split | Safe | Unsafe | Total |
|-------|------|--------|-------|
| Train | 19,328 | 9,444 | 28,772 |
| Val | 3,587 | 2,026 | 5,613 |
| Test | 4,541 | 2,048 | 6,589 |

---

## Quick Start

```bash
# Setup
sudo pacman -S portaudio ffmpeg python-pip    # Arch/CachyOS
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# Download and prepare data
PYTHONPATH=. python safecommute/pipeline/download_datasets.py
PYTHONPATH=. python safecommute/pipeline/download_audioset.py
PYTHONPATH=. python safecommute/pipeline/data_pipeline.py
PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py
PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py
PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py

# Train (best config)
PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 0.5 --noise-inject

# Analyze
PYTHONPATH=. python safecommute/pipeline/analyze.py

# Fine-tune for deployment
PYTHONPATH=. python safecommute/pipeline/finetune.py --environment metro

# Live inference
PYTHONPATH=. python safecommute/pipeline/inference.py
```

---

## Experiment History

7 autonomous experiment cycles were run to improve real-world robustness:

| Cycle | Technique | AUC | Verdict |
|-------|-----------|-----|---------|
| 0 | AST Knowledge Distillation | 0.795 | AUC dropped, accuracy up |
| 1 | Sub-Spectral Normalization | 0.784 | Failed — erases discriminative features |
| 2 | Hard Negative Mining | 0.780 | Failed — redundant with focal loss |
| 3 | Aggressive Mixup | 0.775 | Failed — over-regularization |
| 4 | **Gamma Sweep** | **0.800** | **Breakthrough — gamma=3.0 was the problem** |
| 5 | Gamma=0.5 Tuning | 0.793 | Best deployment balance |
| **6** | **Noise Injection** | **0.804** | **Best overall — first additive improvement** |
| 7 | Noise + Label Smoothing | TBD | In progress |

Key discovery: focal loss gamma=3.0 was catastrophically over-regularized, causing 0% accuracy on safe hard negatives (laughter, crowd, speech). Lowering to gamma=0.5 + adding metro noise injection during training produced the best deployable model.

See `research/experiment_cycles.md` for full details.

---

## Team

- **Alessandro Canonico** -- Project Lead & AI Strategist
- **Fabiola Martignetti** -- Behavioral Data & ML Specialist
- **Robbie Urquhart** -- Machine Learning & Edge Engineer

## License

See [LICENSE](LICENSE) for details.
