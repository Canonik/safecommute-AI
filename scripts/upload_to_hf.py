"""
Upload SafeCommute AI models to HuggingFace Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  # paste your write token

Usage:
    PYTHONPATH=. python scripts/upload_to_hf.py
    PYTHONPATH=. python scripts/upload_to_hf.py --repo-id your-username/safecommute-ai
    PYTHONPATH=. python scripts/upload_to_hf.py --private  # private repo
"""

import os
import sys
import json
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from huggingface_hub import HfApi, create_repo


def build_model_card():
    """Generate README.md model card for HuggingFace."""
    return """---
license: cc-by-nc-4.0
tags:
  - audio-classification
  - sound-event-detection
  - safety
  - edge-deployment
  - pytorch
  - mel-spectrogram
language:
  - en
metrics:
  - accuracy
  - f1
  - roc_auc
pipeline_tag: audio-classification
---

# SafeCommute AI — Edge Audio Threat Detector

Privacy-first binary audio classifier for detecting escalation (screaming, gunshots, glass breaking, explosions) in public spaces. Designed for edge deployment — 7MB, 7ms inference on CPU.

**Raw audio is never stored.** Only non-reconstructible mel spectrograms are processed.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | CNN6 + SE attention + GRU + multi-scale pooling |
| Parameters | 1,829,444 (1.83M) |
| Size | 7.0 MB (float32), 5.0 MB (INT8) |
| Latency | ~7ms CPU inference |
| Input | 3-second audio @ 16kHz → (1, 64, 188) log-mel spectrogram |
| Output | Binary: safe (0) vs unsafe (1) |
| Training data | AudioSet, ESC-50, UrbanSound8K, YouTube, Violence dataset (40,974 samples) |

## Results

| Metric | Test |
|--------|------|
| AUC-ROC | 0.856 |
| Accuracy | 0.675 |
| F1 | 0.684 |

### SOTA Comparison

| Model | Params | AUC | Latency |
|-------|--------|-----|---------|
| **SafeCommute (ours)** | **1.83M** | **0.856** | **12ms** |
| PANNs CNN14 | 81.8M | 0.624 | 250ms |
| AST Transformer | 86.6M | 0.615 | 965ms |

## Files

- `safecommute_edge_model.pth` — Base model (float32, 7MB)
- `safecommute_edge_model_int8.pth` — INT8 quantized (5MB)
- `metro_model.pth` — Fine-tuned for metro deployment
- `metro_thresholds.json` — Optimized detection thresholds for metro
- `config.json` — Model configuration
- `feature_stats.json` — Feature normalization stats (mean/std)

## Usage

```python
import torch
import librosa
import numpy as np

# Load model
from safecommute.model import SafeCommuteCNN
model = SafeCommuteCNN()
model.load_state_dict(torch.load("safecommute_edge_model.pth", map_location="cpu"))
model.eval()

# Load audio
y, sr = librosa.load("audio.wav", sr=16000, mono=True)
y = y[:48000]  # 3 seconds

# Extract features
mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64, n_fft=1024, hop_length=256)
log_mel = librosa.power_to_db(mel, ref=1.0)
tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(tensor)
    prob_unsafe = torch.softmax(logits, dim=1)[0][1].item()
    print(f"Unsafe probability: {prob_unsafe:.3f}")
```

## Personalization

Fine-tune for any deployment environment (metro, bar, bus):

```bash
PYTHONPATH=. python safecommute/pipeline/finetune.py \\
    --environment metro --ambient-dir your_ambient_audio/ --freeze-cnn
```

## Data Strategy

Three-layer approach:
1. **Layer 1** (universal threats): AudioSet screaming, gunshot, explosion, glass breaking + YouTube real screams + violence dataset
2. **Layer 2** (hard negatives): AudioSet laughter, crowd, speech, music, applause, cheering, singing + ESC-50 + UrbanSound8K
3. **Layer 3** (deployment-specific): Recorded ambient audio for target environment (added during fine-tuning)

## Citation

```bibtex
@misc{safecommute2026,
    title={SafeCommute AI: Privacy-Preserving Edge Audio Classifier for Public Safety},
    author={Canonico, Alessandro and Martignetti, Fabiola and Urquhart, Robbie},
    year={2026},
    institution={Bocconi University}
}
```

## License

CC-BY-NC-4.0 (model weights). Code is available at the project repository.
"""


def main():
    parser = argparse.ArgumentParser(description='Upload SafeCommute models to HuggingFace')
    parser.add_argument('--repo-id', type=str, default='Canonik/safecommute-ai',
                        help='HuggingFace repo ID (default: Canonik/safecommute-ai)')
    parser.add_argument('--private', action='store_true',
                        help='Create private repository')
    args = parser.parse_args()

    api = HfApi()

    if not api.token:
        print("ERROR: No HuggingFace token found.")
        print("Run: huggingface-cli login")
        sys.exit(1)

    print("=" * 60)
    print(f" Uploading SafeCommute AI to HuggingFace")
    print(f" Repo: {args.repo_id}")
    print("=" * 60)

    # Create repo
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        print(f"\n  Repository: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"  Repo creation note: {e}")

    # Build config.json
    config = {
        "model_type": "SafeCommuteCNN",
        "architecture": "CNN6 + SE + GRU + multi-scale pooling",
        "n_mels": 64,
        "time_frames": 188,
        "sample_rate": 16000,
        "duration_sec": 3.0,
        "n_classes": 2,
        "labels": ["safe", "unsafe"],
        "parameters": 1829444,
    }

    # Write temp files
    with tempfile.TemporaryDirectory() as tmp:
        # Config
        config_path = os.path.join(tmp, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Model card
        card_path = os.path.join(tmp, "README.md")
        with open(card_path, 'w') as f:
            f.write(build_model_card())

        # Upload model card + config
        print("\n  Uploading model card and config...")
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                        repo_id=args.repo_id)
        api.upload_file(path_or_fileobj=config_path, path_in_repo="config.json",
                        repo_id=args.repo_id)

    # Upload model files
    files_to_upload = [
        ("safecommute_edge_model.pth", "safecommute_edge_model.pth"),
        ("safecommute_edge_model_int8.pth", "safecommute_edge_model_int8.pth"),
        ("models/metro_model.pth", "metro_model.pth"),
        ("models/metro_thresholds.json", "metro_thresholds.json"),
        ("feature_stats.json", "feature_stats.json"),
    ]

    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  Uploading {repo_path} ({size_mb:.1f} MB)...")
            api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path,
                            repo_id=args.repo_id)
        else:
            print(f"  Skipping {local_path} (not found)")

    # Upload model source code
    source_files = [
        "safecommute/model.py",
        "safecommute/constants.py",
        "safecommute/features.py",
        "safecommute/dataset.py",
        "safecommute/utils.py",
    ]
    print("\n  Uploading source code...")
    for src in source_files:
        if os.path.exists(src):
            api.upload_file(path_or_fileobj=src, path_in_repo=src,
                            repo_id=args.repo_id)

    print(f"\n  Done! View at: https://huggingface.co/{args.repo_id}")
    print(f"  Clone with: git clone https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
