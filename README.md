# SafeCommute AI

Research codebase for privacy-preserving edge audio classification. The classifier consumes only PCEN spectrogram tiles, never raw audio, on the bet that those tiles are non-invertible enough to defeat reconstruction attacks from off-the-shelf adversaries. Workshop paper in progress.

Three documents define the current state of the project:

- [paper.md](paper.md): in-progress writeup. Being restructured around the privacy contribution.
- [RESULTS.md](RESULTS.md): every measured detection number with hardware disclosure.
- [VALIDATE_AND_IMPROVE.md](VALIDATE_AND_IMPROVE.md): honest audit of which claims are reproduced and which are still pending.

The non-invertibility claim is currently an assertion, not a measured result. The reconstruction-attack harness (Griffin-Lim plus a pretrained HiFi-GAN baseline, with a mel-vs-PCEN ablation) is planned for `tests/privacy/`; until that lands, treat the privacy hook as a working hypothesis under test.

## How it works

1. **Base model** trained on AudioSet, UrbanSound8K, and ESC-50: detects universal threats (screams, gunshots, glass breaking, violent yelling) and rejects universal hard negatives (laughter, music, sirens).
2. **Per-site fine-tune** on roughly 30 minutes of held-out ambient audio from the target environment; see [safecommute/pipeline/finetune.py](safecommute/pipeline/finetune.py).
3. **Inference** runs on CPU at the edge, 3-second sliding window over PCEN tiles. No GPU required.

## Model

| Metric | Value |
|---|---|
| Architecture | CNN6 + SE + GRU + multi-scale pooling |
| Parameters | 1,829,444 (1.83 M) |
| Size FP32 | 7.00 MB |
| Size INT8 ONNX | 3.72 MB |
| AUC-ROC | 0.804 (measured 2026-04-21, base, prepared_data/test/) |
| CPU latency (current reference) | 4.0 ms FP32 ONNX / 2.8 ms INT8 ONNX, Ryzen 7 7435HS, 8 threads, `torch==2.11.0`, `onnxruntime==1.24.4` |

The historical "around 12 ms" figure was measured on a Ryzen 5 at 1 thread (that hardware is no longer on hand); on the current reference machine the same 1-thread configuration gives 26.9 ms ONNX FP32. Every row is reproducible with `python tests/measure_latency.py`. Full hardware-disclosed table in [RESULTS.md](RESULTS.md).

## Deployment audit (n=1 site, measured 2026-04-22)

The base model is a research artefact, not a shippable classifier without per-site fine-tuning. Held-out evaluation on 19 `youtube_metro_quarantine` wavs (deterministic 50/50 split of a 34-wav quarantine; the other 15 wavs are used only for threshold calibration, never for training):

| Recipe | FP on held-out (n=19) | Threat recall (n=57) |
|---|---|---|
| Default (`--freeze-cnn --keep-safe-ratio 0.5 --epochs 10`) | 38.2 % | 89.5 % |
| Best architecture-preserving tweak (`--keep-safe-ratio 0.1 --epochs 20 --freeze-cnn`) | 29.4 % | 87.7 % |
| Best tweak at FP cap 5 % (threshold 0.710, k=1) | 2.9 % | 49.1 % |
| metro_tweak3 + low_fpr 0.601 + majority-k=2 | **0.0 %** | **78.9 %** |

The 5 % FP target is met on this one site. The 88 % recall target is not; measured recall is 78.9 %, a 9-point gap. Both numbers are at n=19, so the confidence intervals are wide; the planned statistical-rigor pass adds Wilson intervals everywhere. See [paper.md §1.4](paper.md) and [tests/reports/metro_lever_sweep.json](tests/reports/metro_lever_sweep.json) for the full 4-checkpoint × 3-k × 2-threshold matrix.

## Setup

```bash
python -m venv venv && source venv/bin/activate.fish
pip install -r requirements.txt
```

Python 3.11 to 3.14 are known to work; PyTorch 2.0+ required.

## Quick start

```bash
# Live microphone demo on the base model.
PYTHONPATH=. python demo.py

# Fine-tune for a specific environment.
PYTHONPATH=. python safecommute/pipeline/finetune.py \
    --environment my_location --ambient-dir raw_data/my_location/ --freeze-cnn

# Headless inference loop.
PYTHONPATH=. python safecommute/pipeline/inference.py
```

A packaged demo bundle (PyTorch checkpoint plus `infer.py` and a quickstart `short.md`) lives at [demo/safecommute-v2-demo.zip](demo/safecommute-v2-demo.zip).

## Training from scratch

```bash
PYTHONPATH=. python safecommute/pipeline/download_datasets.py
PYTHONPATH=. python safecommute/pipeline/download_audioset.py
PYTHONPATH=. python safecommute/pipeline/data_pipeline.py
PYTHONPATH=. python safecommute/pipeline/prepare_youtube_data.py
PYTHONPATH=. python safecommute/pipeline/prepare_violence_data.py
PYTHONPATH=. python safecommute/pipeline/verify_pipeline.py
PYTHONPATH=. python safecommute/pipeline/train.py --focal --cosine --strong-aug --gamma 0.5 --noise-inject
```

The pipeline splits each source deterministically by SHA-256 of the filename (via [safecommute/utils.py](safecommute/utils.py)) and verifies the absence of cross-split source leakage with [safecommute/pipeline/verify_pipeline.py](safecommute/pipeline/verify_pipeline.py).

## Datasets

Trained on AudioSet, UrbanSound8K, and ESC-50; held-out per-site recordings are used for fine-tune evaluation. Source attributions and download instructions are in the [`safecommute/pipeline/download_*.py`](safecommute/pipeline/) scripts.

## Authors

Alessandro Canonico, Fabiola Martignetti, Robbie Urquhart. Bocconi University, Milan.

## License

See [LICENSE](LICENSE).
