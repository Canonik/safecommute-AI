# SafeCommute AI

Privacy-first edge audio classifier for detecting escalation in public spaces. Built at Bocconi University.

**No raw audio is ever stored.** Only non-reconstructible PCEN spectrograms are processed on-device. GDPR-compliant by architecture.

## How It Works

1. **Base model** detects universal threats: screams, gunshots, glass breaking, violent yelling
2. **Fine-tune** for your environment: record 1-2 hours of ambient audio, run `finetune.py`
3. **Deploy**: real-time inference at 12ms on CPU, no GPU needed

## Model

| Metric | Value |
|--------|-------|
| Architecture | CNN6 + SE + GRU + Multi-Scale Pooling |
| Parameters | 1.83M |
| Size | 7MB |
| CPU Latency | ~12ms |
| AUC-ROC | 0.804 |

See [RESULTS.md](RESULTS.md) for full benchmarks and per-source accuracy.

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

See [DEPLOY.md](DEPLOY.md) for the full deployment guide.

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
