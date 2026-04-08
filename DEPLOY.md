# SafeCommute AI — Deployment Guide

## Overview

SafeCommute is NOT a plug-and-play detector. It requires **per-environment fine-tuning** to be deployable. The base model detects threats (screams, gunshots, glass breaking) but also false-positives on normal speech and crowd noise.

Fine-tuning teaches the model what "safe" sounds like in YOUR specific environment.

## Step 1: Record Ambient Audio

Record 1-2 hours of normal audio from your deployment location:
- Use your phone (voice recorder app, 16kHz mono if possible)
- Capture different conditions: rush hour, quiet periods, announcements, conversations
- Place the phone where the microphone would be deployed
- **All this audio is labeled as SAFE** — it's what "normal" sounds like here

Save the recordings as `.wav` files in a directory, e.g. `raw_data/my_metro/`

## Step 2: Fine-Tune

```bash
PYTHONPATH=. python safecommute/pipeline/finetune.py \
    --environment my_metro \
    --ambient-dir raw_data/my_metro/ \
    --freeze-cnn \
    --epochs 10 \
    --lr 1e-4
```

This will:
- Load the base model (`models/safecommute_v2.pth`)
- Process your ambient audio into 3-second PCEN spectrograms
- Fine-tune the GRU + classifier layers (CNN frozen to prevent forgetting)
- Optimize thresholds on the fine-tuned model
- Save to `models/my_metro_model.pth` + `models/my_metro_thresholds.json`

Takes ~10 minutes on GPU, ~30 minutes on CPU.

## Step 3: Test

```bash
# Run deployment acceptance tests
PYTHONPATH=. python safecommute/pipeline/test_deployment.py \
    --model models/my_metro_model.pth

# Live test with your mic
PYTHONPATH=. python demo.py --model models/my_metro_model.pth
```

## Step 4: Deploy

```bash
# Real-time inference
PYTHONPATH=. python safecommute/pipeline/inference.py
```

The inference engine:
- Captures 1-second audio chunks, sliding 3-second window
- Energy gating: silence = auto-safe
- Temporal smoothing over 4 predictions
- Dual thresholds: amber (warning) / red (alert)
- **No audio is ever saved to disk** (GDPR-compliant)

## Expected Performance After Fine-Tuning

Based on metro fine-tuning tests:
- Threat detection: ~86%
- False positive rate: ~7%
- With threshold optimization: FP rate can be pushed to <5%

## Hardware Requirements

- CPU: any x86_64 or ARM (Raspberry Pi 4+)
- RAM: 512MB minimum
- Storage: 10MB for model
- Microphone: any USB or built-in mic
- No GPU required — 12ms inference on CPU
