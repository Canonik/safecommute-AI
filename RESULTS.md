# SafeCommute AI — Model Results

All numbers below are produced by [`tests/verify_performance_claims.py`](tests/verify_performance_claims.py); every row is re-measurable from `models/safecommute_v2.pth` + `prepared_data/test/`. Hardware-dependent numbers (latency) carry a disclosure line.

## Best Model: safecommute_v2 (gamma=0.5 + noise injection)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.804 |
| Accuracy @ 0.50 | 70.3% |
| F1 weighted @ 0.50 | 0.716 |
| Overall FPR @ 0.50 | 34.2% |
| Parameters | 1,829,444 (1.83M) |
| Size FP32 | 7.00 MB |
| Size INT8 ONNX | 3.72 MB |
| Config | `--focal --cosine --strong-aug --gamma 0.5 --noise-inject` |

## Per-Source Accuracy (all sources, not just the headline four)

### Threat Detection (unsafe class, TPR @ thr=0.50)

| Source | Accuracy | Samples |
|--------|----------|---------|
| viol_violence | 96.3% | 133 |
| as_yell | 90.6% | 286 |
| as_gunshot | 89.5% | 342 |
| as_glass | 79.9% | 299 |
| as_screaming | 79.1% | 235 |
| yt_scream | 78.2% | 225 |
| as_explosion | 74.0% | 219 |
| as_shout | 64.7% | 309 |

### Hard Negatives (safe class, 1 − FPR @ thr=0.50)

| Source | Accuracy | FPR | Samples |
|--------|----------|-----|---------|
| synth_silence | 100.0% | 0.0% | 75 |
| synth_quiet | 100.0% | 0.0% | 75 |
| synth_ambient | 100.0% | 0.0% | 75 |
| as_applause | 88.3% | 11.7% | 264 |
| as_cheering | 87.4% | 12.6% | 380 |
| hns | 86.2% | 13.8% | 643 |
| bg | 78.7% | 21.3% | 582 |
| as_singing | 73.1% | 26.9% | 353 |
| yt_metro | 64.9% | 35.1% | 442 |
| esc | 64.2% | 35.8% | 120 |
| as_music | 62.4% | 37.6% | 484 |
| viol_violence | 39.3% | 60.7% | 163 |
| esc_hns | 37.5% | 62.5% | 40 |
| as_crowd | 42.1% | 57.9% | 423 |
| as_speech | 28.3% | 71.7% | 378 |
| as_laughter | 17.5% | 82.5% | 269 |

## CPU Latency (hardware-disclosed)

| Variant | Median / p99 (ms) | Notes |
|---|---|---|
| ~12 ms on Ryzen 5, 1 core | 12 / — | *Historical*: original marketing figure. Hardware not currently on hand; see footnote. |
| PyTorch eager, Ryzen 7 7435HS, 8T | 8.5 / 9.1 | measured 2026-04-21, `torch.__version__=2.11.0+cu130` |
| ONNX FP32, Ryzen 7 7435HS, 8T | 4.0 / 4.2 | `onnxruntime==1.24.4` |
| ONNX INT8 static, Ryzen 7 7435HS, 8T | 2.8 / 3.2 | Static PTQ via `safecommute/export_quantized.py` |
| ONNX FP32, Ryzen 7 7435HS, 1T | 26.9 / — | single-thread comparable to the original Ryzen 5 claim |
| End-to-end (preprocess + model, INT8, Ryzen 7) | ~50 / — | librosa PCEN dominates (`tests/measure_e2e_latency.py`) |

> **Historical 12 ms footnote**: the headline "~12 ms on Ryzen 5, 1 core" figure was measured on hardware that is no longer on hand. On Ryzen 7 7435HS (the current reference hardware) the 1-thread equivalent is **26.9 ms ONNX FP32**; the 8-thread path reaches **2.8 ms INT8 / 4.0 ms FP32**. All rows are re-runnable with `python tests/measure_latency.py`.

## Leakage / Confusion Matrix @ thr=0.50

|  | predicted safe | predicted unsafe |
|---|---|---|
| actual safe | 65.8% | **34.2%** (leakage) |
| actual unsafe | 19.1% | 80.9% |

> The earlier "41% leakage" figure in pitch materials was a hand-typed literal in [scripts/generate_pitch_figures.py:267-268](scripts/generate_pitch_figures.py#L267-L268) (now model-derived). The measured overall FPR at the default threshold is **34.2%**. The pitch-deck script now reads every number from [`tests/reports/figures_source.json`](tests/reports/figures_source.json).

## Known Limitations

- **Speech false positive rate = 71.7%** (measured) — the base model confuses normal speech with threats because both share spectral features (sustained vowel formants, fricatives overlap with hiss/glass). Fix: per-environment fine-tuning with recorded ambient audio.
- **Laughter FPR = 82.5% > speech FPR** — the docs previously highlighted only speech. Laughter is actually the larger false-alarm source.
- **Not standalone-deployable** — requires per-environment fine-tuning to be usable. See §Deployment for the measured post-fine-tune numbers.
- **Non-violent "violence" subset → 60.7% FPR** — the `viol_violence` safe-side clips bundle non-violent physical interactions that the model confuses with threats.

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
