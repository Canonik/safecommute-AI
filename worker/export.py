"""Inline FP32 → INT8 ONNX export for the worker.

`safecommute/export.py` and `safecommute/export_quantized.py` hardcode the
paper-artefact paths (`models/safecommute_v2*.onnx`), which would let one
job's output clobber another's and overwrite the base checkpoint. This
module does the same two-step export — (.pth → FP32 ONNX, opset 17, single
file) → (static INT8 PTQ, fused, calibrated on real PCEN tensors) — but
with caller-supplied output paths and an in-memory `PCENCalibrationReader`.

Reuses:
  * safecommute.model.SafeCommuteCNN (architecture, immutable)
  * safecommute.constants.{N_MELS, TIME_FRAMES}
  * safecommute.export_quantized._load_test_tensors (calibration corpus)
  * safecommute.export_quantized.PCENCalibrationReader
  * safecommute.export_quantized._fuse_graph
  * safecommute.export_quantized._quantize_static
  * safecommute.export_quantized._validate_int8

Nothing in safecommute/** is modified; this module is a thin binding layer.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import torch

from safecommute.model import SafeCommuteCNN
from safecommute.constants import N_MELS, TIME_FRAMES
from safecommute.export_quantized import (
    PCENCalibrationReader,
    _fuse_graph,
    _load_test_tensors,
    _quantize_static,
    _validate_int8,
)

log = logging.getLogger(__name__)


def _export_fp32_onnx(pth_path: str, out_path: str) -> None:
    """Load `.pth` → export to a single-file FP32 ONNX at `out_path`.

    Mirrors safecommute/export.py:export_onnx semantics (opset 17, no dynamic
    axes, inlined weights), but writes to `out_path` instead of the hardcoded
    `models/safecommute_v2.onnx`.
    """
    model = SafeCommuteCNN()
    model.load_state_dict(torch.load(pth_path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, TIME_FRAMES)
    external_data = out_path + ".data"
    try:
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            opset_version=17,
        )
    try:
        import onnx
        onnx_model = onnx.load(out_path, load_external_data=True)
        if os.path.exists(external_data):
            os.remove(external_data)
        onnx.save(onnx_model, out_path, save_as_external_data=False)
        onnx.checker.check_model(onnx_model)
    except Exception as e:  # noqa: BLE001
        log.warning("ONNX validation/inline failed (file still saved): %s", e)


def export_int8_onnx(pth_path: str, int8_out: str, *,
                     calib_samples: int = 128,
                     per_channel: bool = True) -> Dict[str, float]:
    """Produce a calibrated INT8 ONNX at `int8_out` from a `.pth` checkpoint.

    Returns a dict with file-size + logit-delta stats (same keys as
    safecommute.export_quantized._validate_int8).
    """
    # Scratch FP32 ONNX + fused ONNX live alongside the target INT8 file so
    # callers can wipe the whole artefact dir in one `rm -rf` on failure.
    base = int8_out.rsplit(".onnx", 1)[0]
    fp32_onnx = base + "_fp32.onnx"
    fused_onnx = base + "_fused.onnx"

    import onnxruntime as ort  # noqa: F401 — fail fast with a clear import error

    _export_fp32_onnx(pth_path, fp32_onnx)
    _fuse_graph(fp32_onnx, fused_onnx)

    tensors = _load_test_tensors(calib_samples)
    sess = ort.InferenceSession(fused_onnx, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    del sess

    calib = PCENCalibrationReader(input_name, tensors)
    _quantize_static(fused_onnx, int8_out, calib, per_channel=per_channel)

    stats = _validate_int8(fused_onnx, int8_out, tensors)
    # Clean up the transient FP32 ONNX + fused ONNX — the paying customer
    # only needs the INT8 artefact, not the 3 variants.
    for tmp in (fp32_onnx, fused_onnx):
        if os.path.exists(tmp):
            os.remove(tmp)
    return stats
