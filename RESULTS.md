# SafeCommute AI — Model Results

## Best Model: safecommute_v2 (gamma=0.5 + noise injection)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.804 |
| Accuracy | 70.3% |
| F1 | 0.716 |
| Parameters | 1.83M |
| Size | 7MB float32 |
| CPU Latency | ~12ms |
| Config | `--focal --cosine --strong-aug --gamma 0.5 --noise-inject` |

## Per-Source Accuracy

### Threat Detection (unsafe)
| Source | Accuracy | Samples |
|--------|----------|---------|
| as_yell | 90.6% | 286 |
| as_screaming | 79.1% | 235 |
| yt_scream | 78.2% | 225 |
| as_shout | 64.7% | 309 |

### Hard Negatives (safe)
| Source | Accuracy | Samples |
|--------|----------|---------|
| yt_metro | 64.9% | 442 |
| as_crowd | 42.1% | 423 |
| as_speech | 28.3% | 378 |
| as_laughter | 17.5% | 269 |

## Known Limitations

- **Speech false positive rate ~72%** — the model confuses normal speech with threats because both share spectral features. Fix: per-environment fine-tuning with recorded ambient audio.
- **Laughter/crowd false positives** — improved from 0% (old gamma=3.0) to 42% crowd / 17% laughter, but still high.
- **Not standalone-deployable** — requires per-environment fine-tuning to be usable.

## Experiment Summary

| Experiment | AUC | Outcome |
|------------|-----|---------|
| v2 baseline (gamma=3.0) | 0.856* | Hard negatives at 0% — unusable |
| AST distillation | 0.795 | AUC dropped |
| Sub-spectral norm | 0.784 | Failed |
| Hard negative mining | 0.780 | Failed — redundant with focal loss |
| Aggressive mixup | 0.775 | Failed — over-regularization |
| **Gamma sweep** | **0.800** | **Breakthrough: gamma=3.0 was the problem** |
| Gamma=0.5 | 0.793 | Best deployment balance |
| **Gamma=0.5 + noise inject** | **0.804** | **Best overall** |
| Noise + label smoothing | 0.763 | Incompatible with focal loss |
| LibriSpeech speech data | 0.784 | Failed — data can't fix feature overlap |

*v2 baseline AUC measured on different test snapshot, not directly comparable.

Key discovery: focal loss gamma=3.0 was catastrophically over-regularized. Hard negatives got zero gradient. Lowering to gamma=0.5 + adding environmental noise injection during training produced the best deployable model.
