"""
End-to-end latency measurement: raw PCM → PCEN → normalize → model → softmax.

§5.6 of VALIDATE_AND_IMPROVE.md: the paper cannot claim "12 ms" without
separating model-only from end-to-end latency. librosa PCEN alone takes
~80 ms on a Pi 4 and ~2-5 ms on desktop x86 — preprocessing can dominate
in the deployment budget.

Reports median / p10 / p90 / p99 for three measurements:
  1. preprocessing-only    (raw 3 s PCM → normalized PCEN tensor)
  2. model-only            (PCEN tensor → softmax probability)
  3. total end-to-end      (PCM → probability)

Measures both:
  - PyTorch FP32 eager (the reference)
  - ONNX Runtime FP32 (safecommute_v2.onnx)
  - ONNX Runtime INT8 (safecommute_v2_int8.onnx, if present)

Each run emits hw_disclosure() so the paper's latency table can cite
hardware, threads, BLAS backend, and library versions inline.

Run:
    PYTHONPATH=. python tests/measure_e2e_latency.py
    PYTHONPATH=. python tests/measure_e2e_latency.py --n-measure 200
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.constants import MODEL_SAVE_PATH, SAMPLE_RATE  # noqa: E402
from safecommute.features import preprocess  # noqa: E402
from tests._common import (  # noqa: E402
    hw_disclosure, load_model, load_stats, time_forward,
)


def _make_pcm(seed: int = 42) -> np.ndarray:
    """Deterministic 3 s synthetic PCM that PCEN will happily consume."""
    rng = np.random.default_rng(seed)
    n = SAMPLE_RATE * 3  # 48 000
    # Mix: pink-ish noise + occasional spike, float32 in [-1, 1].
    y = (0.05 * rng.standard_normal(n)).astype(np.float32)
    t = np.arange(n) / SAMPLE_RATE
    y += 0.1 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)
    return y


def _time(label: str, fn: Callable, x, n_warmup: int, n_measure: int) -> Dict:
    s = time_forward(fn, x, n_warmup=n_warmup, n_measure=n_measure)
    print(f"  {label:<38} median={s['median']:>7.2f} ms  "
          f"p10={s['p10']:>7.2f}  p90={s['p90']:>7.2f}  "
          f"p99={s['p99']:>7.2f}  mean={s['mean']:>7.2f}")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-warmup', type=int, default=20)
    p.add_argument('--n-measure', type=int, default=200)
    p.add_argument('--out', default='tests/reports/measure_e2e_latency.json')
    args = p.parse_args()

    pcm = _make_pcm()
    mean, std = load_stats()

    print("SafeCommute AI — end-to-end latency")
    print("=" * 72)
    hw = hw_disclosure()
    print(f"  Hardware: {hw['cpu_model']} ({hw['cores']} cores, "
          f"{hw['torch_threads']} torch threads)  governor={hw['governor']}")
    print(f"  Software: torch {hw['torch_version']}, "
          f"onnxruntime {hw['onnxruntime_version']}, "
          f"numpy {hw['numpy_version']}")
    print()

    results: Dict[str, Dict] = {'hw_disclosure': hw, 'runs': {}}

    # ── 1. Preprocessing only ──────────────────────────────────────────
    def do_preprocess(_pcm):
        preprocess(_pcm, mean, std)

    pre = _time('preprocess (librosa PCEN + norm)',
                do_preprocess, pcm, args.n_warmup, args.n_measure)
    results['runs']['preprocess'] = pre

    # ── 2. Model-only (PyTorch FP32 eager) ─────────────────────────────
    model = load_model(MODEL_SAVE_PATH, torch.device('cpu'))
    x_pre = preprocess(pcm, mean, std)  # shape (1, 1, 64, 188)

    def run_eager(_x):
        with torch.inference_mode():
            logits = model(_x)
            torch.softmax(logits, dim=1)

    pt = _time('model-only (PyTorch FP32 eager)',
               run_eager, x_pre, args.n_warmup, args.n_measure)
    results['runs']['model_pytorch_fp32'] = pt

    # ── 3. Total (PyTorch FP32 eager) ──────────────────────────────────
    def e2e_pt(_pcm):
        t = preprocess(_pcm, mean, std)
        with torch.inference_mode():
            logits = model(t)
            torch.softmax(logits, dim=1)

    e2e_pt_stats = _time('e2e  (preprocess + PyTorch FP32)',
                         e2e_pt, pcm, args.n_warmup, args.n_measure)
    results['runs']['e2e_pytorch_fp32'] = e2e_pt_stats

    # ── 4, 5. ONNX FP32 + e2e ─────────────────────────────────────────
    onnx_fp32 = MODEL_SAVE_PATH.replace('.pth', '.onnx')
    if os.path.exists(onnx_fp32):
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = torch.get_num_threads()
            sess = ort.InferenceSession(
                onnx_fp32, sess_options=so,
                providers=['CPUExecutionProvider'])
            in_name = sess.get_inputs()[0].name
            x_np = x_pre.numpy().astype(np.float32)

            def run_ort(_x):
                sess.run(None, {in_name: x_np})

            ort_stats = _time('model-only (ONNX FP32)',
                              run_ort, x_pre,
                              args.n_warmup, args.n_measure)
            results['runs']['model_onnx_fp32'] = ort_stats

            def e2e_ort(_pcm):
                t = preprocess(_pcm, mean, std).numpy().astype(np.float32)
                sess.run(None, {in_name: t})

            e2e_ort_stats = _time('e2e  (preprocess + ONNX FP32)',
                                  e2e_ort, pcm,
                                  args.n_warmup, args.n_measure)
            results['runs']['e2e_onnx_fp32'] = e2e_ort_stats
        except Exception as e:
            print(f"  ONNX FP32 FAILED: {type(e).__name__}: {e}")
            results['runs']['model_onnx_fp32'] = {'error': str(e)}
    else:
        print(f"  ONNX FP32 skipped ({onnx_fp32} missing — run safecommute/export.py)")

    # ── 6, 7. ONNX INT8 + e2e ─────────────────────────────────────────
    onnx_int8 = MODEL_SAVE_PATH.replace('.pth', '_int8.onnx')
    if os.path.exists(onnx_int8):
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = torch.get_num_threads()
            sess8 = ort.InferenceSession(
                onnx_int8, sess_options=so,
                providers=['CPUExecutionProvider'])
            in_name8 = sess8.get_inputs()[0].name
            x_np = x_pre.numpy().astype(np.float32)

            def run_int8(_x):
                sess8.run(None, {in_name8: x_np})

            q_stats = _time('model-only (ONNX INT8)',
                            run_int8, x_pre,
                            args.n_warmup, args.n_measure)
            results['runs']['model_onnx_int8'] = q_stats

            def e2e_int8(_pcm):
                t = preprocess(_pcm, mean, std).numpy().astype(np.float32)
                sess8.run(None, {in_name8: t})

            e2e_int8_stats = _time('e2e  (preprocess + ONNX INT8)',
                                   e2e_int8, pcm,
                                   args.n_warmup, args.n_measure)
            results['runs']['e2e_onnx_int8'] = e2e_int8_stats
        except Exception as e:
            print(f"  ONNX INT8 FAILED: {type(e).__name__}: {e}")
            results['runs']['model_onnx_int8'] = {'error': str(e)}
    else:
        print(f"  ONNX INT8 skipped ({onnx_int8} missing — run safecommute.export_quantized)")

    # ── Budget analysis ───────────────────────────────────────────────
    if 'e2e_onnx_int8' in results['runs'] and 'preprocess' in results['runs']:
        pre_med = results['runs']['preprocess']['median']
        total = results['runs']['e2e_onnx_int8']['median']
        print()
        print(f"  Budget (ONNX INT8 path):")
        print(f"    preprocess : {pre_med:>6.2f} ms  "
              f"({pre_med/total*100:>5.1f} %)")
        model_only = results['runs']['model_onnx_int8']['median']
        print(f"    model      : {model_only:>6.2f} ms  "
              f"({model_only/total*100:>5.1f} %)")
        print(f"    total e2e  : {total:>6.2f} ms")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"  Report → {args.out}")


if __name__ == '__main__':
    main()
