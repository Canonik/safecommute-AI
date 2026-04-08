# SafeCommute AI — Research

## Current Best Model (Cycle 6)

| Metric | Value |
|--------|-------|
| Architecture | CNN6 + SE + GRU + Multi-Scale Pooling |
| Parameters | 1.83M |
| Config | gamma=0.5 focal + cosine LR + strong aug + noise injection |
| AUC-ROC | 0.804 |
| Accuracy | 70.3% |
| F1 | 0.716 |
| Model Size | 7.0 MB (float32), 5.0 MB (INT8) |
| Inference Latency | ~12ms (CPU) |

## Key Findings

1. **gamma=3.0 was catastrophically over-regularized** — hard negatives (speech 0.3%, laughter 0%, crowd 0%) got zero gradient. Lowering to gamma=0.5 fixed this.
2. **Environmental noise injection works** — mixing metro ambient during training improved crowd accuracy from 22% to 42% and overall AUC from 0.793 to 0.804.
3. **Speech remains the critical failure mode** — 72% FP rate due to insufficient speech training data (~2k clips). Fix: add LibriSpeech/CommonVoice/VoxCeleb.
4. **Training tricks (SSN, HNM, aggressive mixup) all failed at gamma=3.0** — they add regularization that conflicts with already-aggressive focal loss.

## Directory Structure

```
research/
  experiment_cycles.md    # Autonomous experiment loop results (Cycles 0-7)
  experiment_log.md       # Historical experiment results (v1, v2, ablations, LOSO)
  data_sources.md         # Dataset citations and details
  literature_review.md    # 20-paper survey with priority table
  NEXT_STEPS.md           # Publication and deployment roadmap
  figures/                # Publication-quality figures
  results/                # Experiment results (JSON)
  experiments/            # Experiment scripts (ablation, LOSO, CV, etc.)
```

## Experiment History

See [experiment_cycles.md](experiment_cycles.md) for the full 7-cycle autonomous loop.
See [experiment_log.md](experiment_log.md) for historical baselines, ablations, LOSO, and SOTA comparisons.
