# Step 0 Baseline — SafeCommute AI, 2026-04-21

> **Historical snapshot.** The "Blockers observed for later steps" section at the end of this document is fully resolved as of 2026-04-22:
> - ONNX single-file fix → done (see `models/safecommute_v2.onnx`, 6.99 MB with no `.data` sidecar)
> - Static INT8 on Conv → done (`models/safecommute_v2_int8.onnx`, 3.72 MB, AUC Δ = 0.002)
> - 15 ms / 30 ms latency targets → met by both the 8T ONNX FP32 path (3.97 ms) and the INT8 path (2.8 ms)
>
> This document remains in the repo as the reproducibility provenance for the latency numbers cited in `paper.md` §1.2 and `README.md` §CPU Latency. The tables below are authoritative for what was measured on 2026-04-21; current state lives in `tests/reports/SUMMARY.md` + `tests/reports/verify_performance_claims.json`.


Hardware: AMD Ryzen 7 7435HS, 8 cores / 16 threads (8 used by default).
Python 3.14.4, torch 2.11.0+cu130, onnxruntime 1.24.4, librosa 0.11.0, scipy 1.17.1, numpy 2.4.3.
Checkpoint: `models/safecommute_v2.pth` (7 MB FP32).

## Phase A — base model on prepared_data/test/
From [tests/baseline/analyze_model_baseline.txt](analyze_model_baseline.txt).

| Metric | Baseline measured | Doc claim | Match |
|---|---|---|---|
| AUC-ROC | 0.804 | 0.804 ± 0.010 | PASS |
| Accuracy @ 0.50 | 0.703 | 0.703 ± 0.010 | PASS |
| F1 weighted @ 0.50 | 0.716 | 0.716 ± 0.010 | PASS |
| Params | 1,829,444 | 1.83M | PASS |
| FP32 size | 7.00 MB | 7.00 MB | PASS |
| TPR as_yell | 0.906 | 0.906 | PASS |
| TPR as_screaming | 0.791 | 0.791 | PASS |
| TPR yt_scream | 0.782 | 0.782 | PASS |
| TPR as_shout | 0.647 | 0.647 | PASS |
| TPR as_gunshot | 0.893 | not in RESULTS.md | ADD to paper |
| TPR as_explosion | 0.738 | not in RESULTS.md | ADD |
| TPR as_glass | 0.800 | not in RESULTS.md | ADD |
| TPR viol_violence | 0.963 | not in RESULTS.md | ADD |
| FPR yt_metro | 0.351 | 0.351 | PASS |
| FPR as_crowd | 0.579 | 0.579 | PASS |
| FPR as_speech | 0.717 | 0.717 | PASS — the 71.7% headline |
| FPR as_laughter | 0.825 | 0.825 | PASS (larger than speech — docs omit) |
| Micro threat recall | 0.809 | 0.82 ± 0.03 | PASS (edge of band) |
| Overall FPR @ 0.50 | **0.342** | 0.41 (claimed in `generate_pitch_figures.py:267-268`) | **FAIL — 41% is a hand-typed literal; measured is 34.2%** |

## Latency — fresh baseline vs audit doc
From [tests/baseline/measure_latency_baseline.txt](measure_latency_baseline.txt).

| Config | Measured on Ryzen 7 7435HS | Audit doc said | Note |
|---|---|---|---|
| Eager no_grad, 8T | 8.98 ms | 108 ms | Audit out of date |
| Eager inference_mode, 8T | 8.42 ms | — | — |
| inference_mode, 4T | 13.32 ms | — | — |
| inference_mode, 2T | 18.06 ms | — | — |
| inference_mode, 1T | 34.24 ms | — | closest to "Ryzen 5 1T" target |
| JIT+optimize, 1T | 33.46 ms | — | (JIT hit ONNX assertion — cosmetic, number still reported) |
| ONNX CPU, 8T | **3.97 ms** | 65 ms | **Audit out of date** |
| ONNX CPU, 1T | 26.92 ms | 325 ms | Audit out of date |
| Dynamic INT8 quant, 8T | 9.01 ms | — | Linear+GRU only; Conv2d FP32 still |

**Key finding**: the "~12 ms" marketing number is actually reproducible at 8 threads on current hardware (ONNX 3.97 ms, eager 8.42 ms). The audit doc's 108 ms / 65 ms / 325 ms figures reflect an earlier measurement setup that no longer applies. The 12 ms claim is **soft-defensible** on any ≥4-thread x86 CPU with the ONNX path; the 1-thread target still takes ~27 ms (2.25× the claim). Step 11 footnotes will document both.

## Claim-duplication surface, per-source FPR (all 16)
From analyze_model_baseline.txt threshold-sweep / per-source tables:

| Source | n | FPR @ 0.50 | bucket |
|---|---|---|---|
| as_applause | 264 | 0.117 | CLEAN |
| as_cheering | 380 | 0.126 | CLEAN |
| as_crowd | 423 | 0.579 | BAD |
| as_laughter | 269 | 0.825 | BAD (top offender) |
| as_music | 484 | 0.376 | MID |
| as_singing | 353 | 0.269 | MID |
| as_speech | 378 | 0.717 | BAD (the headline) |
| bg | 582 | 0.213 | MID |
| esc | 120 | 0.358 | MID |
| esc_hns | 40 | 0.625 | BAD |
| hns | 643 | 0.138 | CLEAN |
| synth_ambient | 75 | 0.000 | CLEAN |
| synth_quiet | 75 | 0.000 | CLEAN |
| synth_silence | 75 | 0.000 | CLEAN |
| viol_violence | 163 | 0.607 | BAD (safe-side violence data) |
| yt_metro | 442 | 0.351 | MID |

## Threshold sweep (for `low_fpr` threshold design discussion)

| thr | FPR | TPR | BalAcc |
|---|---|---|---|
| 0.10 | 0.875 | 0.998 | 0.561 |
| 0.20 | 0.689 | 0.978 | 0.644 |
| 0.30 | 0.585 | 0.954 | 0.685 |
| 0.40 | 0.458 | 0.892 | 0.717 |
| **0.50** | **0.342** | **0.809** | **0.733** |
| 0.60 | 0.216 | 0.646 | 0.715 |
| 0.70 | 0.093 | 0.375 | 0.641 |
| 0.80 | 0.012 | 0.098 | 0.543 |
| 0.90 | 0.000 | 0.006 | 0.503 |

To push FPR ≤ 5% on the base model, threshold ≈ 0.70+ is required, which slashes TPR to 37.5%. The `low_fpr` threshold from `finetune.py` is what makes the deployment gate work *after* per-site fine-tuning.

## Blockers observed for later steps
- Step 1 (ONNX single-file fix): `models/safecommute_v2.onnx` on disk is 100 KB + a 7 MB `.data` sidecar; confirms the audit finding.
- Step 5 (static INT8 on Conv): dynamic INT8 (Linear+GRU only) already achieves 9.01 ms on 8T — so the FP32 ONNX at 3.97 ms is actually *faster*. Static-INT8 on Conv may not save further latency on this CPU but is still required for the size target (≤ 6 MB → currently `safecommute_v2_int8.pth` is 5.3 MB which satisfies the size gate, but that's the wrong format for the demo bundle).
- Latency targets in `test_deployment.py:18-19` (mean ≤ 15 ms, p99 ≤ 30 ms) are already *met* by the ONNX 8T path. The 1T target is the harder one.

## What moves into the paper unchanged
- Every Phase A row is reproducible from the current checkpoint.
- γ=0.5 AUC = 0.804 confirmed.
- Speech FP ≈ 72% confirmed.
- Laughter FP > speech FP (82.5% > 71.7%) is a **new honest addition** to the paper — the doc currently highlights only speech.
