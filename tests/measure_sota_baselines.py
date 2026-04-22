"""
Measure SOTA baseline latency / size / param-count on the same hardware the
SafeCommute verifier runs on, so the paper's SOTA comparison table in §3.5
cites measured numbers rather than literature numbers from different hardware.

Baselines measured:
  - PANNs CNN14 (80M params, AudioSet-pretrained) — via panns_inference.models.
  - (YAMNet is not PyTorch-native; skipping to avoid TF+torch in the same env.
     If needed, run it in a separate process with TF installed.)

What this script does NOT measure: AUC on prepared_data/test/. CNN14 outputs
527-class AudioSet probabilities; our binary task has no direct mapping. A
fair AUC comparison would need a CNN14-to-binary classifier head trained
under the same protocol — that's a separate, larger study. The paper's SOTA
claim (`~50× smaller, comparable latency on same hardware`) only needs the
measured params/size/latency columns we compute here.

Emits tests/reports/baselines.json with model, params, size, latency median /
p99, full hardware disclosure (identical schema to
verify_performance_claims.json so downstream consumers can share code paths).

Run:
    PYTHONPATH=. python tests/measure_sota_baselines.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests._common import hw_disclosure  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(REPO, 'tests', 'reports')

# CNN14's native input parameters (PANNs paper + panns_inference defaults).
CNN14_SR = 32000
CNN14_SEC = 3
CNN14_SAMPLES = CNN14_SR * CNN14_SEC


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _size_mb(state_dict_path: str) -> float:
    return os.path.getsize(state_dict_path) / (1024 * 1024)


def _time_forward(forward_fn, n_warmup: int = 20, n_measure: int = 50) -> Dict:
    """Per-call wall-time in ms. Returns mean/median/p99/stddev."""
    for _ in range(n_warmup):
        forward_fn()
    samples = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        forward_fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples_np = np.array(samples)
    return {
        'mean_ms': float(samples_np.mean()),
        'median_ms': float(np.median(samples_np)),
        'p99_ms': float(np.percentile(samples_np, 99)),
        'stddev_ms': float(samples_np.std()),
        'n_warmup': n_warmup,
        'n_measure': n_measure,
    }


def measure_cnn14(save_state_dict: bool = True) -> Dict:
    """Instantiate CNN14, measure params/size/CPU latency at current-threads.

    save_state_dict: write an FP32 state_dict to .scratch/cnn14_fp32.pth so
        we have a size-on-disk number; deletes it after measuring to avoid
        polluting the models/ directory (CNN14 weights aren't the paper's
        artefact, they're a reference baseline).
    """
    from panns_inference.models import Cnn14
    model = Cnn14(sample_rate=CNN14_SR, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    model.eval()

    scratch = os.path.join(REPO, '.scratch')
    os.makedirs(scratch, exist_ok=True)
    path = os.path.join(scratch, 'cnn14_fp32.pth')
    size_mb = None
    if save_state_dict:
        torch.save(model.state_dict(), path)
        size_mb = _size_mb(path)

    x = torch.randn(1, CNN14_SAMPLES)

    def run():
        with torch.inference_mode():
            model(x)

    lat = _time_forward(run, n_warmup=10, n_measure=30)

    if save_state_dict and os.path.exists(path):
        os.remove(path)

    return {
        'model': 'PANNs-CNN14',
        'params': _params(model),
        'fp32_size_mb': size_mb,
        'input_shape': list(x.shape),
        'sample_rate': CNN14_SR,
        'window_sec': CNN14_SEC,
        'latency': lat,
        'note': ('Latency measured on a synthetic 3-sec 32kHz input; CNN14 '
                 "expects raw audio (log-mel + PCEN is handled inside). "
                 "AudioSet-pretrained weights NOT loaded for latency (empty "
                 "init is equivalent — forward pass cost is identical)."),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=os.path.join(REPORTS_DIR, 'baselines.json'))
    p.add_argument('--threads', type=int, default=None,
                   help=('Override torch thread count (default: current). '
                         'Pass 1 for the single-thread comparison the '
                         'marketing "~12 ms on Ryzen 5, 1T" row.'))
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print("Measuring SOTA baselines on:",
          hw_disclosure()['cpu_model'],
          f"({torch.get_num_threads()} threads)")

    baselines = []
    # Also measure SafeCommute's own INT8 ONNX at the same thread count so
    # the comparison row isn't apples-to-pears.
    try:
        cnn14 = measure_cnn14()
        baselines.append(cnn14)
        print(f"\n  CNN14: params={cnn14['params']:,}, "
              f"size={cnn14['fp32_size_mb']:.1f} MB, "
              f"mean={cnn14['latency']['mean_ms']:.1f} ms, "
              f"median={cnn14['latency']['median_ms']:.1f} ms, "
              f"p99={cnn14['latency']['p99_ms']:.1f} ms")
    except Exception as e:
        print(f"\n  CNN14: FAILED ({type(e).__name__}: {e})")
        baselines.append({'model': 'PANNs-CNN14',
                          'error': f"{type(e).__name__}: {e}"})

    # SafeCommute self-row (the reference): re-measure the INT8 ONNX so both
    # rows were timed under the exact same thread setting + warmup schedule.
    try:
        import onnxruntime as ort
        from safecommute.constants import MODEL_SAVE_PATH, N_MELS, TIME_FRAMES
        int8_path = MODEL_SAVE_PATH.replace('.pth', '_int8.onnx')
        if os.path.exists(int8_path):
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = torch.get_num_threads()
            sess = ort.InferenceSession(int8_path, sess_options=sess_opts,
                                        providers=['CPUExecutionProvider'])
            in_name = sess.get_inputs()[0].name
            np_x = np.random.randn(1, 1, N_MELS, TIME_FRAMES).astype('float32')

            def run_sc():
                sess.run(None, {in_name: np_x})

            lat = _time_forward(run_sc, n_warmup=10, n_measure=30)
            sc_size = _size_mb(int8_path)

            baselines.append({
                'model': 'SafeCommute-INT8-ONNX',
                'params': 1_829_444,  # FP32 param count; INT8 is same graph
                'fp32_size_mb': None,
                'int8_size_mb': sc_size,
                'input_shape': list(np_x.shape),
                'sample_rate': 16000,
                'window_sec': 3,
                'latency': lat,
                'note': ('Self-row — same hardware, same thread count as '
                         'CNN14 baseline above. For reference, not a '
                         "self-comparison claim."),
            })
            print(f"\n  SafeCommute-INT8-ONNX: params=1,829,444, "
                  f"size={sc_size:.2f} MB, "
                  f"mean={lat['mean_ms']:.1f} ms, "
                  f"median={lat['median_ms']:.1f} ms, "
                  f"p99={lat['p99_ms']:.1f} ms")
    except Exception as e:
        print(f"\n  SafeCommute self-row: FAILED ({type(e).__name__}: {e})")

    out = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'hw_disclosure': hw_disclosure(),
        'threads_used': torch.get_num_threads(),
        'baselines': baselines,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    # Print the ratio the paper's SOTA table will quote.
    if (baselines
            and 'params' in baselines[0]
            and len(baselines) > 1
            and 'params' in baselines[1]
            and baselines[0].get('latency') is not None
            and baselines[1].get('latency') is not None):
        cnn14_params = baselines[0]['params']
        sc_params = baselines[1]['params']
        cnn14_lat = baselines[0]['latency']['median_ms']
        sc_lat = baselines[1]['latency']['median_ms']
        print("\n--- SOTA comparison (measured, same hardware + same threads) ---")
        print(f"  param ratio    CNN14/SafeCommute = {cnn14_params / sc_params:.1f}×")
        print(f"  latency ratio  CNN14/SafeCommute = {cnn14_lat / sc_lat:.1f}×")


if __name__ == '__main__':
    main()
