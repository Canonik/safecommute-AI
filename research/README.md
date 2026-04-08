# SafeCommute AI — Research

## Current Deployment Candidate (Cycle 6, provisional)

| Metric | Value |
|--------|-------|
| Architecture | CNN6 + SE + GRU + Multi-Scale Pooling |
| Parameters | 1.83M |
| AUC-ROC (current champion) | 0.804 |
| Accuracy (current champion) | 0.703 |
| F1 (current champion) | 0.716 |
| Model Size | 7.0 MB (float32), 5.0 MB (INT8) |
| Inference Latency | ~7ms (CPU) |
| Training data | 28,772 train / 5,613 val / 6,589 test (40,974 total) |

## Data Strategy

Three-layer approach:
1. **Layer 1 — Universal threats** (unsafe): AudioSet screaming, shout, yell, gunshot, explosion, glass breaking + YouTube real screams + violence dataset
2. **Layer 2 — Hard negatives** (safe): AudioSet laughter, crowd, speech, music, applause, cheering, singing + ESC-50 + UrbanSound8K
3. **Layer 3 — Deployment ambient** (safe, fine-tuning only): per-environment recorded audio

Dropped: CREMA-D, SAVEE, TESS, RAVDESS (acted speech, 35-52% accuracy)

## Key Results

### SOTA Comparison
| Model | Params | AUC | Latency |
|-------|--------|-----|---------|
| **SafeCommute** | **1.83M** | **0.856** | **12ms** |
| PANNs CNN14 | 81.8M | 0.624 | 250ms |
| AST | 86.6M | 0.615 | 965ms |

### Reliability Key Finding
Laughter (6%), crowd (11%), speech (20%) are essential hard negatives — without them the model classifies all loud sounds as threats. Threat sounds generalize well across sources (gunshot 90%, screaming 81%).

### Reliability-First Protocol
- Freeze one immutable benchmark set and prevent silent benchmark drift.
- Evaluate a hard-negative stress suite (laughter/crowd/speech-heavy).
- Gate runs on hard-negative FPR, per-threat recall floors, and worst-source floor.
- Track deployment KPIs: alerts/hour on ambient-only audio, nuisance alert rate, threat miss rate.

### Deployment
Metro fine-tuning: AUC 0.837 -> 0.867. With optimized threshold (0.666): 86% threat detection, 6.9% FP rate.

## Directory Structure

```
research/
  README.md                    # This file
  NEXT_STEPS.md                # Publication roadmap
  experiment_log.md            # All experiment results (append-only)
  data_sources.md              # Dataset citations and details
  literature_review.md         # Paper survey
  figures/                     # Publication-quality figures
  results/                     # Experiment results (JSON + model checkpoints)
  experiments/                 # Experiment scripts
    eval_utils.py              # Shared evaluation utilities
    loso_evaluation.py         # Leave-one-source-out
    ablation_study.py          # Architectural ablation
    cross_validation.py        # 5-fold CV
    threshold_optimization.py  # Threshold tuning
    reliability_protocol.py    # Reliability gates + immutable benchmark checks
    constrained_model_selection.py # Select candidates under shout/crowd/metro constraints
    [+ 12 more experiment scripts]
  generate_paper_figures.py    # Publication figure generator
  run_all_experiments.py       # Master experiment runner
```

See [experiment_log.md](experiment_log.md) for all results.
See [NEXT_STEPS.md](NEXT_STEPS.md) for the publication roadmap.
