# SafeCommute AI 🚇🎧

## Privacy-First Audio AI for Early Detection of Escalation in Public Transport

SafeCommute AI is an on-device machine learning system designed to detect early acoustic signs of escalation in crowded public transport environments, such as aggressive shouting or distress screams.

Built by students at Bocconi University, the system is designed around a privacy-first, edge-only architecture. All inference runs locally on the device, helping reduce regulatory and privacy concerns by ensuring that raw audio is never recorded, stored, or transmitted.

---

## 🎯 The Problem and Our Solution

### The Challenge
Security personnel in public transport hubs must monitor large, noisy, high-density spaces with limited attention and limited real-time visibility.

### The Barrier
Traditional CCTV systems are mostly reactive, often useful only after an event has already occurred. They can also feel intrusive and raise serious privacy and GDPR concerns.

### The SafeCommute Advantage
SafeCommute AI offers a proactive and privacy-preserving alternative.

- Runs continuously on a 3-second sliding audio window
- Processes data directly on edge devices
- Converts audio into non-reconstructible Mel-spectrograms
- Never records, stores, or transmits raw audio
- Outputs only a compact binary alert: `Safe` or `Unsafe`

This allows transport staff to respond faster while preserving passenger privacy.

---

## 👥 Team

- **Alessandro Canonico** — Project Lead & AI Strategist
- **Fabiola Martignetti** — Behavioral Data & ML Specialist
- **Robbie Urquhart** — Machine Learning & Edge Engineer

---

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** PyTorch or TensorFlow / Keras
- **Signal Processing:** Librosa, NumPy
- **Audio I/O:** PyAudio
- **Deployment Hardware:** Raspberry Pi, laptops, or Android devices

---

## 💻 Environment Setup Guide

### 🐧 CachyOS / Arch Linux

Since CachyOS often uses Fish shell and PipeWire, follow these steps:

#### 1. Install system dependencies
```bash
sudo pacman -S portaudio ffmpeg python-pip
```

#### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate.fish
```

#### 3. Install python requirements
```bash
pip install -r requirements.txt
```

### 🪟 Windows
```bash
1. Install Python
```

Make sure Python 3.10 or later is installed.

During installation, check the box:

Add Python to PATH
2. Open PowerShell as Administrator
3. Create and activate a virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
4. Install dependencies
```bash
pip install -r requirements.txt
```

If pyaudio fails to install, you may need Microsoft C++ Build Tools or a precompiled wheel.


## 📖 Technical Deep Dive
### 1. Privacy by Design

SafeCommute AI is built around strict privacy constraints.

Feature Extraction: Audio is converted into Mel-spectrograms, preserving acoustic intensity patterns while removing intelligible speech content.
Edge Processing: All inference runs locally on the device.
Minimal Output: No raw audio leaves the device. The system emits only a final risk classification.

This design reduces privacy exposure and supports deployment in regulation-sensitive settings.

### 2. Sliding Window Engine

The model processes audio in real time using a rolling analysis window.

Window size: 3 seconds
Stride: 1 second
Smoothing logic: Alerts are triggered only after multiple high-risk windows occur consecutively

This helps reduce false positives while preserving responsiveness.

### 3. Training Logic: Safe vs Unsafe

SafeCommute AI uses a binary classifier for clarity and operational simplicity.

#### Class 0 — Safe

Includes:

normal conversation
laughter
transit background noise
station announcements

#### Class 1 — Unsafe

Includes:

aggressive shouting
distress screams
intense verbal escalation


## 🚀 How to Use SafeCommute AI
### Step 1: Prepare the data

Run the dataset preparation script:
```bash
python prepare_all_data.py
```

This script downloads and mixes datasets such as RAVDESS and UrbanSound8K, combining clean vocal distress signals with noisy transit-style backgrounds to better simulate real-world stations.

### Step 2: Train the model

Fine-tune the model locally:
```bash
python train_model.py
```
This generates the trained model file:

safecommute_edge_model.pth
### Step 3: Run live inference

Start the real-time MVP monitor:
```bash
python mvp_inference.py
```

# Troubleshooting:
If the risk score remains stuck around 0.47, your microphone is probably muted, disconnected, or not properly selected in your operating system settings.

## 📈 Next Steps Toward Production

To turn this MVP into a stakeholder-ready system for operators such as ATM or Trenord, the next milestones are:

- Acquire more "loud but safe" data
- Gather examples such as cheering crowds, station announcements, and non-threatening public noise to reduce false alarms.
- Acoustic environment mapping
- Calibrate the model for specific stations and transport settings, including echo-heavy metro platforms like those in Milan.
- Edge hardware deployment
- Port and optimize the model for Raspberry Pi-based field testing.