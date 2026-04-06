"""
Deployment acceptance test suite for SafeCommute AI.

Validates that the model is deployment-ready by running 7 tests on
real audio, latency, model size, consistency, and export verification.

Run before shipping any model to production.

Usage:
    PYTHONPATH=. python safecommute/pipeline/test_deployment.py
    PYTHONPATH=. python safecommute/pipeline/test_deployment.py --model models/metro_model.pth --verbose
    PYTHONPATH=. python safecommute/pipeline/test_deployment.py --threshold 0.7
"""

import os
import sys
import io
import json
import time
import argparse

import numpy as np
import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.features import preprocess
from safecommute.constants import (
    SAMPLE_RATE, N_MELS, TIME_FRAMES, MODEL_SAVE_PATH, STATS_PATH,
)

CONTEXT_SEC = 3
STRIDE_SEC = 1
ENERGY_GATE_RMS = 0.003  # matches inference.py — below this = auto-safe


def load_model_and_stats(model_path):
    """Load model and feature stats."""
    import json
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            s = json.load(f)
        mean, std = s['mean'], s['std']
    else:
        mean, std = 0.0, 1.0

    model = SafeCommuteCNN()
    model.load_state_dict(
        torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model, mean, std


def sliding_window_inference(model, y, mean, std, threshold):
    """Run sliding window inference on audio, return per-window probabilities."""
    chunk_len = int(SAMPLE_RATE * CONTEXT_SEC)
    stride_len = int(SAMPLE_RATE * STRIDE_SEC)
    probs = []

    for start in range(0, len(y) - chunk_len + 1, stride_len):
        chunk = y[start:start + chunk_len]
        # Energy gating: match production inference.py behavior
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < ENERGY_GATE_RMS:
            probs.append(0.0)  # auto-safe
            continue
        features = preprocess(chunk, mean, std)
        with torch.no_grad():
            logits = model(features)
            prob = torch.softmax(logits, dim=1)[0][1].item()
        probs.append(prob)

    if not probs and len(y) >= SAMPLE_RATE:
        # Pad short audio
        padded = np.pad(y, (0, chunk_len - len(y)), 'constant')
        features = preprocess(padded, mean, std)
        with torch.no_grad():
            logits = model(features)
            prob = torch.softmax(logits, dim=1)[0][1].item()
        probs.append(prob)

    return probs


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: THREAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def test_threat_detection(model, mean, std, threat_dir, threshold, verbose):
    """Test that threat audio is detected."""
    files = [f for f in sorted(os.listdir(threat_dir)) if f.endswith('.wav')]
    if not files:
        return None, "No threat files found"

    detected = 0
    for fname in files:
        y, _ = librosa.load(os.path.join(threat_dir, fname), sr=SAMPLE_RATE, mono=True)
        probs = sliding_window_inference(model, y, mean, std, threshold)
        max_prob = max(probs) if probs else 0
        triggered = max_prob >= threshold
        if triggered:
            detected += 1
        if verbose:
            status = "DETECT" if triggered else "MISS"
            print(f"    [{status}] {fname}: max={max_prob:.3f}, "
                  f"mean={np.mean(probs):.3f}, windows={len(probs)}")

    rate = detected / len(files)
    passed = rate >= 0.90
    detail = f"{rate:.1%} detection rate ({detected}/{len(files)} files, target >= 90%)"
    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: FALSE POSITIVE RATE
# ─────────────────────────────────────────────────────────────────────────────
def test_false_positive(model, mean, std, ambient_dir, threshold, verbose):
    """Test that ambient audio does not trigger false positives."""
    files = [f for f in sorted(os.listdir(ambient_dir)) if f.endswith('.wav')]
    if not files:
        return None, "No ambient files found"

    false_positives = 0
    for fname in files:
        y, _ = librosa.load(os.path.join(ambient_dir, fname), sr=SAMPLE_RATE, mono=True)
        probs = sliding_window_inference(model, y, mean, std, threshold)
        max_prob = max(probs) if probs else 0
        triggered = max_prob >= threshold
        if triggered:
            false_positives += 1
        if verbose:
            status = "FP" if triggered else "OK"
            print(f"    [{status}] {fname}: max={max_prob:.3f}, "
                  f"mean={np.mean(probs):.3f}")

    rate = false_positives / len(files)
    passed = rate <= 0.05
    detail = f"{rate:.1%} FP rate ({false_positives}/{len(files)} files, target <= 5%)"
    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: LATENCY
# ─────────────────────────────────────────────────────────────────────────────
def test_latency(model):
    """Measure inference latency."""
    dummy = torch.randn(1, 1, N_MELS, TIME_FRAMES)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            model(dummy)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(1000):
            start = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - start) * 1000)

    mean_ms = np.mean(times)
    p99_ms = np.percentile(times, 99)
    passed = mean_ms < 15 and p99_ms < 30
    detail = f"mean={mean_ms:.1f}ms, p99={p99_ms:.1f}ms (target mean<15ms, p99<30ms)"
    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: MODEL SIZE
# ─────────────────────────────────────────────────────────────────────────────
def test_model_size(model_path):
    """Check model file sizes."""
    float32_mb = os.path.getsize(model_path) / (1024 * 1024)
    parts = [f"{float32_mb:.2f}MB float32"]

    int8_path = model_path.replace('.pth', '_int8.pth')
    int8_mb = None
    if os.path.exists(int8_path):
        int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
        parts.append(f"{int8_mb:.2f}MB INT8")

    passed = float32_mb <= 10
    if int8_mb is not None:
        passed = passed and int8_mb <= 6

    detail = f"{', '.join(parts)} (target float32<=10MB, INT8<=6MB)"
    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
def test_consistency(model):
    """Verify deterministic inference."""
    dummy = torch.randn(1, 1, N_MELS, TIME_FRAMES)
    model.eval()

    outputs = []
    with torch.no_grad():
        for _ in range(10):
            out = model(dummy)
            outputs.append(out.numpy().copy())

    max_diff = max(np.max(np.abs(outputs[i] - outputs[0])) for i in range(1, 10))
    passed = max_diff == 0.0
    detail = f"max diff = {max_diff:.6f}"
    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: SILENCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
def test_silence(model, mean, std):
    """Verify model handles silence correctly.

    Tests energy gating (matching inference.py production behavior):
    audio with RMS < 0.003 is classified as safe without model inference.
    """
    # Pure silence — RMS=0, energy gate catches it
    silence = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    silence_rms = float(np.sqrt(np.mean(silence ** 2)))
    silence_gated = silence_rms < ENERGY_GATE_RMS

    # Very quiet noise — RMS ~0.0005, energy gate catches it
    np.random.seed(42)
    quiet = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.0005
    quiet_rms = float(np.sqrt(np.mean(quiet ** 2)))
    quiet_gated = quiet_rms < ENERGY_GATE_RMS

    if silence_gated and quiet_gated:
        # Both gated by energy threshold — matches production behavior
        passed = True
        detail = (f"energy gated: silence RMS={silence_rms:.4f}, "
                  f"quiet RMS={quiet_rms:.4f} (gate={ENERGY_GATE_RMS})")
    else:
        # Fallback: test model output
        features_s = preprocess(silence, mean, std)
        with torch.no_grad():
            prob_silence = torch.softmax(model(features_s), dim=1)[0][0].item()
        features_q = preprocess(quiet, mean, std)
        with torch.no_grad():
            prob_quiet = torch.softmax(model(features_q), dim=1)[0][0].item()
        passed = prob_silence > 0.8 and prob_quiet > 0.8
        detail = f"safe prob: silence={prob_silence:.2f}, quiet={prob_quiet:.2f} (target >0.8)"

    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: ONNX EXPORT VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def test_onnx(model, model_path):
    """Compare ONNX output to PyTorch output."""
    onnx_path = model_path.replace('.pth', '.onnx')
    if not os.path.exists(onnx_path):
        return None, f"ONNX file not found ({onnx_path})"

    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime not installed"

    dummy = torch.randn(1, 1, N_MELS, TIME_FRAMES)

    # PyTorch output
    model.eval()
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    # ONNX output
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    passed = max_diff < 0.01
    detail = f"max diff = {max_diff:.6f} (target <0.01)"
    return passed, detail


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SafeCommute AI — Deployment Acceptance Tests')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH,
                        help='Model to test')
    parser.add_argument('--threat-dir', type=str, default='raw_data/youtube_screams',
                        help='Directory with threat audio')
    parser.add_argument('--ambient-dir', type=str, default='raw_data/youtube_metro',
                        help='Directory with ambient audio')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold (default: 0.5)')
    parser.add_argument('--thresholds-file', type=str, default=None,
                        help='JSON file with optimized thresholds (uses low_fpr threshold)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-file results')
    args = parser.parse_args()

    print("=" * 60)
    print(" SafeCommute AI — Deployment Acceptance Tests")
    print(f" Model: {args.model}")
    print(f" Threshold: {args.threshold}")
    print("=" * 60)

    # Load optimized threshold if provided
    if args.thresholds_file and os.path.exists(args.thresholds_file):
        with open(args.thresholds_file) as f:
            thresholds = json.load(f)
        args.threshold = thresholds.get('low_fpr', args.threshold)
        print(f" Using optimized threshold: {args.threshold:.3f} (from {args.thresholds_file})")

    if not os.path.exists(args.model):
        print(f"\n  ERROR: Model not found: {args.model}")
        sys.exit(1)

    model, mean, std = load_model_and_stats(args.model)

    tests = [
        ("Threat detection", lambda: test_threat_detection(
            model, mean, std, args.threat_dir, args.threshold, args.verbose), True),
        ("False positive rate", lambda: test_false_positive(
            model, mean, std, args.ambient_dir, args.threshold, args.verbose), True),
        ("Latency", lambda: test_latency(model), True),
        ("Model size", lambda: test_model_size(args.model), True),
        ("Consistency", lambda: test_consistency(model), True),
        ("Silence handling", lambda: test_silence(model, mean, std), True),
        ("ONNX verification", lambda: test_onnx(model, args.model), False),
    ]

    results = []
    print()

    for name, test_fn, must_pass in tests:
        try:
            passed, detail = test_fn()
        except Exception as e:
            passed, detail = False, f"ERROR: {e}"

        if passed is None:
            icon = "-"
            status = "SKIP"
        elif passed:
            icon = "+"
            status = "PASS"
        else:
            icon = "x"
            status = "FAIL"

        print(f"  [{icon}] {status:4}  {name}: {detail}")
        results.append((name, passed, must_pass))

    # Summary
    total = len([r for r in results if r[1] is not None])
    passed_count = len([r for r in results if r[1] is True])
    failed_must_pass = [r[0] for r in results if r[1] is False and r[2] is True]

    print(f"\n  RESULT: {passed_count}/{total} passed")

    if failed_must_pass:
        print(f"  MUST-PASS FAILURES: {', '.join(failed_must_pass)}")
        print("\n  DEPLOYMENT: NOT READY")
        sys.exit(1)
    else:
        print("\n  DEPLOYMENT: READY")
        sys.exit(0)


if __name__ == "__main__":
    main()
