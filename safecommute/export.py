"""
Model export and optimization for edge deployment.

Usage:
    python -m safecommute.export

Produces:
    safecommute_edge_model_int8.pth   — INT8 dynamically quantized
    safecommute_edge_model.onnx       — ONNX format (opset 17)
    safecommute_edge_model_scripted.pt — TorchScript traced

Also benchmarks inference latency for each variant.
"""

import os
import sys
import time
import io

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.model import SafeCommuteCNN
from safecommute.constants import MODEL_SAVE_PATH, N_MELS, TIME_FRAMES


def load_model(device='cpu'):
    model = SafeCommuteCNN()
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print(f"Loaded weights from {MODEL_SAVE_PATH}")
    else:
        print(f"Warning: {MODEL_SAVE_PATH} not found. Using random weights.")
    model.eval()
    return model


def measure_size_mb(model):
    """Measure serialized model size in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def measure_latency(model, dummy_input, n_warmup=20, n_runs=200):
    """Measure inference latency on CPU."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy_input)
            times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times), np.percentile(times, 99)


def export_int8(model):
    """INT8 dynamic quantization of Linear and GRU layers."""
    print("\n── INT8 Dynamic Quantization ──")
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU}, dtype=torch.qint8)

    path = MODEL_SAVE_PATH.replace('.pth', '_int8.pth')
    torch.save(quantized.state_dict(), path)
    print(f"  Saved → {path}")
    return quantized, path


def export_onnx(model, dummy_input):
    """Export to ONNX format."""
    print("\n── ONNX Export ──")
    path = MODEL_SAVE_PATH.replace('.pth', '.onnx')
    try:
        torch.onnx.export(
            model, dummy_input, path,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            dynamic_axes={"mel_spectrogram": {0: "batch"}},
            opset_version=17,
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved → {path} ({size_mb:.2f} MB)")

        try:
            import onnx
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            print("  ONNX model validated OK")
        except ImportError:
            print("  (onnx package not installed — skipping validation)")

        return path
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return None


def export_torchscript(model, dummy_input):
    """Export via TorchScript tracing."""
    print("\n── TorchScript Export ──")
    path = MODEL_SAVE_PATH.replace('.pth', '_scripted.pt')
    try:
        scripted = torch.jit.trace(model, dummy_input)
        scripted.save(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved → {path} ({size_mb:.2f} MB)")
        return path
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
        return None


def main():
    device = torch.device("cpu")
    dummy = torch.randn(1, 1, N_MELS, TIME_FRAMES)

    # Load original model
    model = load_model(device)
    param_count = sum(p.numel() for p in model.parameters())
    orig_size = measure_size_mb(model)

    print(f"\nOriginal model: {param_count:,} params, {orig_size:.2f} MB")

    # Benchmark original
    mean_ms, std_ms, p99_ms = measure_latency(model, dummy)
    print(f"  Latency (CPU): {mean_ms:.1f} ± {std_ms:.1f} ms  (p99: {p99_ms:.1f} ms)")

    # INT8 quantization
    q_model, q_path = export_int8(model)
    q_size = measure_size_mb(q_model)
    q_mean, q_std, q_p99 = measure_latency(q_model, dummy)
    print(f"  Size: {q_size:.2f} MB  ({(1 - q_size/orig_size)*100:.0f}% reduction)")
    print(f"  Latency (CPU): {q_mean:.1f} ± {q_std:.1f} ms  (p99: {q_p99:.1f} ms)")

    # ONNX
    onnx_path = export_onnx(model, dummy)

    # TorchScript
    ts_path = export_torchscript(model, dummy)

    # ONNX Runtime benchmark (if available)
    if onnx_path:
        try:
            import onnxruntime as ort
            print("\n── ONNX Runtime Benchmark ──")
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            dummy_np = dummy.numpy()

            for _ in range(20):
                sess.run(None, {input_name: dummy_np})

            times = []
            for _ in range(200):
                start = time.perf_counter()
                sess.run(None, {input_name: dummy_np})
                times.append((time.perf_counter() - start) * 1000)

            ort_mean = np.mean(times)
            ort_p99 = np.percentile(times, 99)
            print(f"  Latency: {ort_mean:.1f} ms  (p99: {ort_p99:.1f} ms)")
        except ImportError:
            print("\n  (onnxruntime not installed — skipping ONNX benchmark)")

    # Summary table
    print("\n" + "=" * 65)
    print(" EXPORT SUMMARY")
    print("=" * 65)
    print(f"{'Variant':<25} {'Size (MB)':>10} {'Latency (ms)':>13} {'p99 (ms)':>10}")
    print("-" * 65)
    print(f"{'Original (float32)':<25} {orig_size:>10.2f} {mean_ms:>13.1f} {p99_ms:>10.1f}")
    print(f"{'INT8 Quantized':<25} {q_size:>10.2f} {q_mean:>13.1f} {q_p99:>10.1f}")
    if onnx_path:
        onnx_mb = os.path.getsize(onnx_path) / (1024**2)
        print(f"{'ONNX':<25} {onnx_mb:>10.2f} {'—':>13} {'—':>10}")
    if ts_path:
        ts_mb = os.path.getsize(ts_path) / (1024**2)
        print(f"{'TorchScript':<25} {ts_mb:>10.2f} {'—':>13} {'—':>10}")


if __name__ == "__main__":
    main()
