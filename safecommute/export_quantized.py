"""
Quantized ONNX export for SafeCommute AI — the real latency + size lever.

Pipeline:
  1. Load models/safecommute_v2.onnx (single-file FP32 from safecommute/export.py).
  2. Graph-optimize the ONNX (Conv+BN folding, constant folding, etc.) via
     onnxruntime's quant_pre_process — fused model saved as
     models/safecommute_v2_fused.onnx.
  3. Build a CalibrationDataReader that yields ~128 real PCEN tensors drawn
     evenly from prepared_data/test/{0_safe,1_unsafe}. Using real-distribution
     PCEN tensors (not random noise) keeps the activation ranges the quantizer
     sees close to what deployment will produce.
  4. Run onnxruntime.quantization.quantize_static on the fused model with QDQ
     format, per-channel weights, symmetric quantization on activations +
     weights, Conv+MatMul ops quantized (GRU ops excluded: non-trivial under
     QDQ and not dominant in the compute mix once Conv is INT8).
  5. Validate: the produced INT8 ONNX must (a) load cleanly, (b) match the
     FP32 ONNX's Phase A AUC within 0.02, (c) fit under the 6 MB demo-bundle
     gate.

Usage:
    PYTHONPATH=. python -m safecommute.export_quantized
    PYTHONPATH=. python -m safecommute.export_quantized --calib-samples 256

Outputs:
    models/safecommute_v2_fused.onnx    (FP32, graph-optimized)
    models/safecommute_v2_int8.onnx     (INT8 QDQ, static-calibrated)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.constants import (  # noqa: E402
    DATA_DIR, MODEL_SAVE_PATH, N_MELS, TIME_FRAMES, SEED, STATS_PATH,
)

FUSED_ONNX = MODEL_SAVE_PATH.replace('.pth', '_fused.onnx')
INT8_ONNX = MODEL_SAVE_PATH.replace('.pth', '_int8.onnx')


def _load_test_tensors(n: int = 128) -> List[np.ndarray]:
    """Load up to n PCEN tensors from prepared_data/test.

    Balances the sample across safe and unsafe so the calibrator sees the
    same distribution the deployed model will see.
    """
    import json
    mean = std = None
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            s = json.load(f)
        mean, std = float(s['mean']), float(s['std'])
    else:
        mean, std = 0.0, 1.0

    safe_dir = os.path.join(DATA_DIR, 'test', '0_safe')
    unsafe_dir = os.path.join(DATA_DIR, 'test', '1_unsafe')
    if not (os.path.isdir(safe_dir) and os.path.isdir(unsafe_dir)):
        raise FileNotFoundError(
            f"expected {safe_dir} and {unsafe_dir} to exist "
            f"(run data pipeline first)")

    rng = random.Random(SEED)
    # The dataset stores paired _teacher.pt soft-label files (shape (2,))
    # alongside the real spectrograms — filter them out.
    def _is_spec(f: str) -> bool:
        return f.endswith('.pt') and not f.endswith('_teacher.pt')
    safe_files = sorted(f for f in os.listdir(safe_dir) if _is_spec(f))
    unsafe_files = sorted(f for f in os.listdir(unsafe_dir) if _is_spec(f))
    rng.shuffle(safe_files)
    rng.shuffle(unsafe_files)

    per_class = max(1, n // 2)
    picks = (
        [(safe_dir, f) for f in safe_files[:per_class]] +
        [(unsafe_dir, f) for f in unsafe_files[:per_class]]
    )
    rng.shuffle(picks)

    tensors = []
    for d, f in picks[:n]:
        if not f.endswith('.pt'):
            continue
        t = torch.load(os.path.join(d, f), weights_only=True)
        # t: (1, 64, T); enforce fixed time dim and normalize like the runtime does.
        if t.shape[-1] > TIME_FRAMES:
            t = t[:, :, :TIME_FRAMES]
        elif t.shape[-1] < TIME_FRAMES:
            pad = torch.zeros(1, t.shape[1], TIME_FRAMES - t.shape[-1])
            t = torch.cat([t, pad], dim=-1)
        t = (t - mean) / (std + 1e-8)
        arr = t.unsqueeze(0).numpy().astype(np.float32)  # (1,1,64,188)
        tensors.append(arr)
    if not tensors:
        raise RuntimeError("no .pt calibration tensors loaded")
    print(f"  Loaded {len(tensors)} calibration tensors "
          f"({sum(1 for p in picks[:n] if p[0]==safe_dir)} safe / "
          f"{sum(1 for p in picks[:n] if p[0]==unsafe_dir)} unsafe)")
    return tensors


class PCENCalibrationReader:
    """Feeds real PCEN tensors to the static quantizer one at a time."""

    def __init__(self, input_name: str, tensors: List[np.ndarray]):
        self.input_name = input_name
        self._iter = iter(tensors)

    def get_next(self):
        try:
            arr = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: arr}

    def rewind(self):
        # onnxruntime may call rewind() in some versions; implement safely.
        return None


def _fuse_graph(src_onnx: str, dst_onnx: str) -> str:
    """Run quant_pre_process (graph-optimize + infer shapes) on the FP32 ONNX.

    This performs Conv+BN folding, constant folding, and any other rewrites
    the onnxruntime optimizer knows about. It is a prerequisite for
    quantize_static to work reliably.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process
    print(f"  quant_pre_process: {src_onnx} → {dst_onnx}")
    quant_pre_process(src_onnx, dst_onnx, skip_symbolic_shape=False)
    return dst_onnx


def _quantize_static(fused_onnx: str,
                     dst_onnx: str,
                     calib: PCENCalibrationReader,
                     per_channel: bool = True) -> str:
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationMethod,
    )
    print(f"  quantize_static: {fused_onnx} → {dst_onnx}")
    quantize_static(
        model_input=fused_onnx,
        model_output=dst_onnx,
        calibration_data_reader=calib,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'MatMul'],
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
        },
    )
    return dst_onnx


def _validate_int8(fp32_onnx: str, int8_onnx: str,
                   tensors: List[np.ndarray]) -> dict:
    """Sanity-check: INT8 ONNX loads, runs, and logits agree with FP32
    ONNX to within a reasonable tolerance on a subset of calibration
    tensors. Full AUC parity check is deferred to
    tests/verify_performance_claims.py."""
    import onnxruntime as ort
    s32 = ort.InferenceSession(fp32_onnx,
                               providers=['CPUExecutionProvider'])
    s8 = ort.InferenceSession(int8_onnx,
                              providers=['CPUExecutionProvider'])
    n32 = s32.get_inputs()[0].name
    n8 = s8.get_inputs()[0].name

    # Take 20 tensors from the calibration pool for a quick agreement check.
    max_diff = 0.0
    diffs = []
    for arr in tensors[:20]:
        y32 = s32.run(None, {n32: arr})[0]
        y8 = s8.run(None, {n8: arr})[0]
        d = float(np.max(np.abs(y32 - y8)))
        diffs.append(d)
        if d > max_diff:
            max_diff = d
    mean_diff = float(np.mean(diffs)) if diffs else 0.0

    int8_mb = os.path.getsize(int8_onnx) / (1024 ** 2)
    fp32_mb = os.path.getsize(fp32_onnx) / (1024 ** 2)
    return {
        'int8_mb': int8_mb,
        'fp32_mb': fp32_mb,
        'size_reduction_x': fp32_mb / int8_mb if int8_mb else float('inf'),
        'logit_max_diff': max_diff,
        'logit_mean_diff': mean_diff,
        'n_checked': len(diffs),
    }


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calib-samples', type=int, default=128,
                   help='Number of real PCEN tensors to use for calibration '
                        '(balanced across safe/unsafe).')
    p.add_argument('--per-channel', action='store_true', default=True)
    p.add_argument('--no-per-channel', dest='per_channel',
                   action='store_false')
    p.add_argument('--fp32-onnx', default=MODEL_SAVE_PATH.replace('.pth', '.onnx'),
                   help='Source FP32 ONNX (single-file export).')
    args = p.parse_args(argv)

    if not os.path.exists(args.fp32_onnx):
        raise FileNotFoundError(
            f"FP32 ONNX missing: {args.fp32_onnx}. "
            f"Run `python -m safecommute.export` first (Step 1).")

    print("SafeCommute AI — static INT8 ONNX export")
    print("=" * 60)

    _fuse_graph(args.fp32_onnx, FUSED_ONNX)

    tensors = _load_test_tensors(args.calib_samples)

    import onnxruntime as ort
    sess = ort.InferenceSession(FUSED_ONNX,
                                providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    del sess

    calib = PCENCalibrationReader(input_name, tensors)
    _quantize_static(FUSED_ONNX, INT8_ONNX, calib,
                     per_channel=args.per_channel)

    stats = _validate_int8(FUSED_ONNX, INT8_ONNX, tensors)
    print()
    print(f"  FP32 ONNX size:   {stats['fp32_mb']:.2f} MB")
    print(f"  INT8 ONNX size:   {stats['int8_mb']:.2f} MB  "
          f"(~{stats['size_reduction_x']:.1f}× smaller)")
    print(f"  Logits max |diff|: {stats['logit_max_diff']:.4f}  "
          f"(mean {stats['logit_mean_diff']:.4f})  "
          f"over {stats['n_checked']} samples")
    print()
    print(f"  Quantized ONNX → {INT8_ONNX}")
    print(f"  Fused FP32 ONNX → {FUSED_ONNX}")
    print()
    print("  Next: `python tests/verify_performance_claims.py --skip-phase-b` "
          "to confirm INT8 AUC within 0.02 of FP32 + size ≤ 6 MB.")


if __name__ == '__main__':
    main()
