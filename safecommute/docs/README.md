# SafeCommute AI — Documentation

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

1. **gamma=3.0 was catastrophically over-regularized** — hard negatives got zero gradient
2. **Environmental noise injection works** — crowd accuracy 22% to 42%
3. **Speech remains the critical failure mode** — 72% FP rate, needs more data
4. **Training tricks fail when focal loss is aggressive** — SSN, HNM, mixup all conflicted with gamma=3.0

## Directory Structure

```
safecommute/docs/
  experiment_cycles.md    # Autonomous experiment loop (Cycles 0-7)
  experiment_log.md       # Historical results (v1, v2, ablations, LOSO)
  data_sources.md         # Dataset citations
  literature_review.md    # 20-paper survey
  NEXT_STEPS.md           # Roadmap
  results/                # JSON result files
  figures/                # Publication figures

safecommute/experiments/
  ablation_study.py, loso_evaluation.py, cross_validation.py, ...
  generate_paper_figures.py, run_all_experiments.py
```

See [experiment_cycles.md](experiment_cycles.md) for the full 7-cycle experiment loop.
See [experiment_log.md](experiment_log.md) for historical baselines and LOSO analysis.
