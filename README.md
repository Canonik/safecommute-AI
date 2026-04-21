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

## Deployment State (measured 2026-04-21, n=1 site)

The base model is a research artefact, not a shippable classifier. On truly-held-out site ambient (34 `youtube_metro_quarantine` clips the fine-tune never sees), **the post-fine-tune ≤ 5 % FP deployment gate is not met**:

| Fine-tune recipe | FP on held-out | Threat recall |
|---|---|---|
| Default (`--freeze-cnn --keep-safe-ratio 0.5 --epochs 10`) | **38.2 %** | 89.5 % |
| Best architecture-preserving tweak (`--keep-safe-ratio 0.1 --epochs 20 --freeze-cnn`) | **29.4 %** | 87.7 % |
| Best tweak at FP ≤ 5 % (threshold 0.710) | 2.9 % | **49.1 %** |

Unfreezing the CNN during fine-tune is *actively worse*. The ≤ 5 % / ≥ 86 % target is unreachable on the 58-clip metro fine-tune pool via training-side re-weighting alone.

Remaining architecture-preserving levers not yet implemented (see [paper.md §7](paper.md)): (a) calibrate `low_fpr` on site-ambient rather than the universal test set, (b) temporal-majority aggregation in [safecommute/pipeline/test_deployment.py](safecommute/pipeline/test_deployment.py) (require ≥ 2 consecutive over-threshold windows), (c) ≥ 5× more field-recorded ambient per site. The honest current state is documented in [paper.md](paper.md), [tests/reports/SUMMARY.md](tests/reports/SUMMARY.md), and [tests/reports/tweak_finetune.json](tests/reports/tweak_finetune.json).

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
