# SafeCommute AI

Privacy-first edge audio classifier for detecting escalation in public spaces. Built at Bocconi University.

**On-device inference is PCEN-only** — the transform applied before the model is non-invertible by construction. Raw audio is never persisted by [safecommute/pipeline/inference.py](safecommute/pipeline/inference.py); the rolling 3-second buffer lives in RAM.

## How It Works

1. **Base model** detects universal threats: screams, gunshots, glass breaking, violent yelling
2. **Fine-tune** for your environment: record 30+ min of ambient audio, run `finetune.py`
3. **Deploy**: real-time CPU inference (see latency table below), no GPU needed

## Model

| Metric | Value |
|--------|-------|
| Architecture | CNN6 + SE + GRU + Multi-Scale Pooling |
| Parameters | 1,829,444 (1.83M) |
| Size FP32 | 7.00 MB |
| Size INT8 ONNX | 3.72 MB |
| AUC-ROC | 0.804 (measured 2026-04-21) |
| CPU Latency | ~12 ms *historical* / 4.0 ms FP32 ONNX / 2.8 ms INT8 ONNX *current reference* |

**Latency disclosure**: the "~12 ms" figure was measured on a Ryzen 5 at 1 thread (original marketing setup, hardware no longer on hand). Current reference hardware is AMD Ryzen 7 7435HS at 8 threads, `torch==2.11.0`, `onnxruntime==1.24.4`; it measures **4.0 ms FP32 / 2.8 ms INT8 ONNX** (medians). On 1 thread the same reference hardware gives **26.9 ms ONNX FP32**. Every row is re-runnable with `python tests/measure_latency.py`. See [RESULTS.md](RESULTS.md) for the full hardware-disclosed table.

See [RESULTS.md](RESULTS.md) for full benchmarks and per-source accuracy, and run [`tests/verify_performance_claims.py`](tests/verify_performance_claims.py) to reproduce every number.

## Deployment State (measured 2026-04-22, n=1 site)

The base model is a research artefact, not a shippable classifier without per-site fine-tuning. Held-out evaluation on 19 `youtube_metro_quarantine` wavs (50/50 deterministic split of the 34-wav quarantine; the 15-wav half is used only for threshold calibration, never for training):

### Phase 1 of the deployment-gap audit (training-side tweaks only, k=1):

| Fine-tune recipe | FP on held-out | Threat recall |
|---|---|---|
| Default (`--freeze-cnn --keep-safe-ratio 0.5 --epochs 10`) | **38.2 %** | 89.5 % |
| Best architecture-preserving tweak (`--keep-safe-ratio 0.1 --epochs 20 --freeze-cnn`) | **29.4 %** | 87.7 % |
| Best tweak at FP ≤ 5 % (threshold 0.710, k=1) | 2.9 % | **49.1 %** |

Unfreezing the CNN during fine-tune is *actively worse*. Training-side re-weighting alone hits an FP plateau at ~29 % on the 58-clip metro fine-tune pool.

### Phase 2 of the audit — architecture-preserving post-hoc levers (k=2):

| Lever applied | FP on held-out (19 wavs) | Threat recall (57 wavs) |
|---|---|---|
| metro_tweak3 + `low_fpr` 0.601 + **majority-k=2** | **0.0 %** | **78.9 %** |

**The ≤ 5 % FP target is now met.** The ≥ 88 % recall target is not — measured recall is 78.9 %, a 9 pt gap vs the 88 % target stated in the fine-tune protocol. Closing that gap requires either (a) ≥ 30 calibration wavs so the site-threshold sweep stops over-tightening, (b) per-event-class threshold calibration, or (c) a learned aggregator on top of window probabilities. See [paper.md §1.4 tweak 5](paper.md) and [tests/reports/metro_lever_sweep.json](tests/reports/metro_lever_sweep.json) for the full 4 ckpts × 3 k × 2 threshold-choice matrix. [paper.md §0](paper.md) + [tests/reports/SUMMARY.md](tests/reports/SUMMARY.md) are the two authoritative numeric sources.

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt

# Live demo (base model)
PYTHONPATH=. python demo.py

# Fine-tune for your environment
PYTHONPATH=. python safecommute/pipeline/finetune.py \
    --environment my_location --ambient-dir raw_data/my_location/ --freeze-cnn

# Deploy
PYTHONPATH=. python safecommute/pipeline/inference.py
```

DEPLOY.md was removed; deployment content is now folded into [RESULTS.md](RESULTS.md) §Deployment and [paper.md](paper.md) §4. The gate is enforced by [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py).

## Training From Scratch

```bash
PYTHONPATH=. python safecommute/pipeline/download_datasets.py
PYTHONPATH=. python safecommute/pipeline/download_audioset.py
PYTHONPATH=. python safecommute/pipeline/data_pipeline.py
PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py
PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py
PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py
PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 0.5 --noise-inject
```

## Team

- **Alessandro Canonico** -- Project Lead & AI Strategist
- **Fabiola Martignetti** -- Behavioral Data & ML Specialist
- **Robbie Urquhart** -- Machine Learning & Edge Engineer

## License

See [LICENSE](LICENSE).
