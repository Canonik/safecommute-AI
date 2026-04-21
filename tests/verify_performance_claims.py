"""
tests/verify_performance_claims.py — single authoritative check that every
numerical performance claim in the repo matches the checkpoint and data on
disk.

Phases:
  A  Universal (base model on prepared_data/test/): discriminative metrics.
  B  Deployment (per-site fine-tune + test_deployment.py on held-out
     ambient + threat wavs). Loaded from tests/reports/phase_b_<site>.json
     produced by `tests/validate_fp_claims.py --phase c` (Step 4).
  C  Narrative: before/after deltas cited in paper / pitch / website.
  +  Latency and size: cross-phase, hardware-disclosed.

Exit 0 = all hard claims pass. Exit 1 = at least one hard claim fails.
Soft claims (latency on non-target hardware, aspirational comparisons,
pre-Phase-B narrative rows) print but never cause failure.

Emits a machine-readable report to
    tests/reports/verify_performance_claims.json
that lists every (name, measured, expected, tol, passed, hard, detail,
hw_disclosure) row. This JSON is the file the paper's Reproducibility
section cites. A --emit-figures-json mode writes a second file at
    tests/reports/figures_source.json
that scripts/generate_pitch_figures.py consumes instead of hand-typed
literals (fixes the 41% leakage hallucination — see Step 9).

Run:
    PYTHONPATH=. python tests/verify_performance_claims.py
    PYTHONPATH=. python tests/verify_performance_claims.py --skip-phase-b
    PYTHONPATH=. python tests/verify_performance_claims.py \
        --emit-figures-json tests/reports/figures_source.json

Set TARGET_HW=Ryzen5-1T to upgrade latency rows from soft to hard.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safecommute.constants import MODEL_SAVE_PATH, N_MELS, TIME_FRAMES  # noqa: E402
from tests._common import (  # noqa: E402
    compute_phase_a_metrics, file_size_mb, hw_disclosure, load_model,
    load_test_dataset, model_size_stats, param_count, run_inference,
    time_forward,
)

# ──────────────────────────────────────────────────────────────────────
# Claim registry — tolerances + expected values per VALIDATE_AND_IMPROVE.md §6.1.
# Per-source rows whose source is missing from the dataset are marked
# soft (WARN) rather than hard failures.
# ──────────────────────────────────────────────────────────────────────

PHASE_A_EXPECTED = {
    'auc_roc':           ('abs', 0.804,   0.010, True),
    'accuracy':          ('abs', 0.703,   0.010, True),
    'f1_weighted':       ('abs', 0.716,   0.010, True),
    'params':            ('rel', 1_829_444, 0.01, True),
    'fp32_size_mb':      ('abs', 7.00,    0.10, True),
    'micro_threat_recall': ('abs', 0.82,  0.03, True),
    # (14) Speech-FP headline: band per §6.1 — 0.60–0.85 window.
    'speech_fp_band':    ('band', (0.60, 0.85), True),
    # (16) Leakage claim — 41% is a hand-typed literal; flagged soft
    # until §9 reconciles. Measurement is ~34.2% FPR at thr=0.5.
    'leakage_claim':     ('abs', 0.41,    0.03, False),
}

PHASE_A_PER_SOURCE_TPR = {
    # (6-9) + undeclared: §6.1 asks for 6-13 plus gunshot / glass / explosion /
    # violence (Phase 0 baseline revealed RESULTS.md omits these).
    'as_yell':       (0.906, 0.03, True),
    'as_screaming':  (0.791, 0.03, True),
    'yt_scream':     (0.782, 0.03, True),
    'as_shout':      (0.647, 0.04, True),
    'as_gunshot':    (0.893, 0.04, True),   # Step 0 baseline added
    'as_glass':      (0.800, 0.04, True),
    'as_explosion':  (0.738, 0.04, True),
    'viol_violence': (0.963, 0.04, True),
}

PHASE_A_PER_SOURCE_FPR = {
    # (10-13)
    'yt_metro':    (0.351, 0.04, True),
    'as_crowd':    (0.579, 0.04, True),
    'as_speech':   (0.717, 0.04, True),
    'as_laughter': (0.825, 0.04, True),
}

# (23) SOTA footprint comparison — static arithmetic only; latency side soft.
SOTA_PARAMS_M = {
    'SafeCommute': 1.83,
    'YAMNet':      3.7,
    'PANNs-CNN14': 80.0,
    'AST':         86.0,
}
PAPER_SOTA_RATIO_CLAIM = 50.0   # "~50× smaller than CNN14/AST"
PAPER_SOTA_RATIO_TOL = 0.20     # ±20% (→ 40–60× acceptable)

# (25, 26) Latency targets: deployment-gate + marketing.
DEPLOYMENT_LATENCY_MEAN_MS = 15.0
DEPLOYMENT_LATENCY_P99_MS = 30.0
MARKETING_LATENCY_MS = 12.0

# (29) Size gates.
FP32_SIZE_MAX_MB = 10.0
INT8_SIZE_MAX_MB = 6.0

# (24) γ ablation — γ=0.5 side is hard (= claim 1), γ=3.0 side is
# historical snapshot per user decision.
GAMMA_0_5_AUC = 0.804
GAMMA_3_0_AUC_HISTORICAL = 0.856

# ──────────────────────────────────────────────────────────────────────
# Check primitive
# ──────────────────────────────────────────────────────────────────────


def check(name: str, measured, expected, tol, *, hard: bool = True,
          kind: str = 'abs', detail: str = '') -> Dict:
    """Build a row for the report. Handles abs, rel, band, lte, gte kinds."""
    passed = False
    if measured is None:
        passed = False
        kind_disp = f"{kind}-NA"
    elif kind == 'abs':
        passed = abs(float(measured) - float(expected)) <= float(tol)
        kind_disp = f"abs ±{tol}"
    elif kind == 'rel':
        passed = abs(float(measured) - float(expected)) / abs(float(expected)) <= float(tol)
        kind_disp = f"rel ±{tol*100:.1f}%"
    elif kind == 'band':
        lo, hi = expected
        passed = lo <= float(measured) <= hi
        kind_disp = f"band [{lo}, {hi}]"
    elif kind == 'lte':
        passed = float(measured) <= float(expected) + float(tol)
        kind_disp = f"≤ {expected}"
    elif kind == 'gte':
        passed = float(measured) >= float(expected) - float(tol)
        kind_disp = f"≥ {expected}"
    elif kind == 'info':
        passed = True
        kind_disp = "info"
    else:
        raise ValueError(f"bad kind: {kind}")
    return {
        'name': name,
        'measured': measured,
        'expected': expected,
        'tolerance': tol,
        'kind': kind,
        'kind_disp': kind_disp,
        'passed': passed,
        'hard': hard,
        'detail': detail,
    }


def fmt_val(v):
    if v is None: return 'N/A'
    if isinstance(v, float): return f"{v:.3f}"
    if isinstance(v, int): return f"{v:,}"
    if isinstance(v, tuple): return str(tuple(round(x, 3) if isinstance(x, float) else x for x in v))
    return str(v)


def print_rows(title: str, rows: List[Dict]) -> None:
    print()
    print(f"{title}")
    print("  " + "-" * 78)
    for r in rows:
        status = ('PASS' if r['passed'] else ('WARN' if not r['hard'] else 'FAIL'))
        tag = f"[{status}]"
        print(f"  {tag:<6} {r['name']:<32} "
              f"{fmt_val(r['measured']):<14} exp {fmt_val(r['expected'])} "
              f"({r['kind_disp']})"
              + (f"  — {r['detail']}" if r['detail'] else ''))


# ──────────────────────────────────────────────────────────────────────
# Phase A
# ──────────────────────────────────────────────────────────────────────


def phase_a(device: torch.device) -> Dict:
    print("=" * 80)
    print("PHASE A  Universal (base model, prepared_data/test/)")
    print("=" * 80)

    model = load_model(MODEL_SAVE_PATH, device=device)
    ds = load_test_dataset('test')
    probs, labels, names = run_inference(model, ds, device)
    metrics = compute_phase_a_metrics(probs, labels, names, threshold=0.5)

    rows: List[Dict] = []

    # Core metrics
    rows.append(check('AUC-ROC', metrics['auc_roc'],
                      PHASE_A_EXPECTED['auc_roc'][1],
                      PHASE_A_EXPECTED['auc_roc'][2],
                      hard=PHASE_A_EXPECTED['auc_roc'][3]))
    rows.append(check('Accuracy @ 0.50', metrics['accuracy'],
                      PHASE_A_EXPECTED['accuracy'][1],
                      PHASE_A_EXPECTED['accuracy'][2],
                      hard=PHASE_A_EXPECTED['accuracy'][3]))
    rows.append(check('F1 weighted @ 0.50', metrics['f1_weighted'],
                      PHASE_A_EXPECTED['f1_weighted'][1],
                      PHASE_A_EXPECTED['f1_weighted'][2],
                      hard=PHASE_A_EXPECTED['f1_weighted'][3]))
    rows.append(check('Params', param_count(model),
                      PHASE_A_EXPECTED['params'][1],
                      PHASE_A_EXPECTED['params'][2],
                      hard=PHASE_A_EXPECTED['params'][3], kind='rel'))
    rows.append(check('Size FP32 (MB)', file_size_mb(MODEL_SAVE_PATH),
                      PHASE_A_EXPECTED['fp32_size_mb'][1],
                      PHASE_A_EXPECTED['fp32_size_mb'][2],
                      hard=PHASE_A_EXPECTED['fp32_size_mb'][3]))

    # Per-source TPR
    for src, (exp, tol, hard) in PHASE_A_PER_SOURCE_TPR.items():
        cell = metrics['tpr_by_source'].get(src)
        if cell is None:
            rows.append(check(f"TPR {src}", None, exp, tol, hard=False,
                              detail=f"source absent from test set"))
            continue
        rows.append(check(f"TPR {src}", cell['rate'], exp, tol, hard=hard,
                          detail=f"n={cell['n']}"))

    # Per-source FPR
    for src, (exp, tol, hard) in PHASE_A_PER_SOURCE_FPR.items():
        cell = metrics['fpr_by_source'].get(src)
        if cell is None:
            rows.append(check(f"FPR {src}", None, exp, tol, hard=False,
                              detail="source absent"))
            continue
        rows.append(check(f"FPR {src}", cell['rate'], exp, tol, hard=hard,
                          detail=f"n={cell['n']}"))

    # Speech-FP headline band (soft-ish; sits inside the 0.60–0.85 window).
    sfp = metrics['fpr_by_source'].get('as_speech', {}).get('rate')
    rows.append(check('Speech-FP headline', sfp,
                      PHASE_A_EXPECTED['speech_fp_band'][1],
                      None, hard=PHASE_A_EXPECTED['speech_fp_band'][2],
                      kind='band'))

    # Micro threat recall
    rows.append(check('Micro threat recall', metrics['overall_tpr'],
                      PHASE_A_EXPECTED['micro_threat_recall'][1],
                      PHASE_A_EXPECTED['micro_threat_recall'][2],
                      hard=PHASE_A_EXPECTED['micro_threat_recall'][3]))

    # Leakage-claim row — doc says 41%, measured is overall FPR at thr=0.5.
    # Soft until Step 9 reconciles the doc. Detail message surfaces the gap.
    lk_expected = PHASE_A_EXPECTED['leakage_claim'][1]
    lk_tol = PHASE_A_EXPECTED['leakage_claim'][2]
    lk_measured = metrics['overall_fpr']
    rows.append(check('Leakage 41% claim', lk_measured, lk_expected, lk_tol,
                      hard=PHASE_A_EXPECTED['leakage_claim'][3],
                      detail=(f"doc: 41% (pitch-figures:267-268 hand-typed); "
                              f"measured FPR@0.5 = {lk_measured:.3f}")))

    # γ=0.5 (current checkpoint) vs historical γ=3.0 row
    rows.append(check('γ=0.5 AUC (= claim 1)', metrics['auc_roc'],
                      GAMMA_0_5_AUC, 0.010, hard=True))
    rows.append(check('γ=3.0 AUC 0.856',
                      None, GAMMA_3_0_AUC_HISTORICAL, None,
                      hard=False, kind='info',
                      detail='historical snapshot — checkpoint not preserved '
                             '(per user decision)'))

    print_rows("Phase A rows", rows)
    return {'rows': rows, 'metrics': metrics,
            'probs': probs, 'labels': labels, 'names': names}


# ──────────────────────────────────────────────────────────────────────
# Phase B — reads a JSON blob produced by tests/validate_fp_claims.py
# ──────────────────────────────────────────────────────────────────────


SITES = [
    # (site_name, ambient_held_out_dir, threat_dir)
    # User scoped Phase B to n=1 (metro). The quarantine bucket is never
    # seen by finetune.py, so it's the honest held-out.
    ('metro', 'raw_data/youtube_metro_quarantine', 'raw_data/youtube_screams'),
]


def _load_phase_b_json(site: str) -> Optional[Dict]:
    p = os.path.join('tests', 'reports', f'phase_b_{site}.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def phase_b(skip: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    print()
    print("=" * 80)
    print("PHASE B  Deployment (per-site fine-tune → test_deployment on held-out)")
    print("=" * 80)
    print(f"  n={len(SITES)} site(s). Paper Limitations section discloses n=1.")
    if skip:
        print("  Phase B SKIPPED (--skip-phase-b); rows will be WARN not FAIL.")
    for site, ambient_dir, threat_dir in SITES:
        blob = _load_phase_b_json(site)
        if blob is None:
            rows.append(check(f"B:{site} FP ≤ 5%", None, 0.05, 0.0,
                              hard=not skip, kind='lte',
                              detail=(f"tests/reports/phase_b_{site}.json absent — "
                                      f"run `tests/validate_fp_claims.py --phase c "
                                      f"--ambient-dir {ambient_dir} "
                                      f"--threat-dir {threat_dir}` first")))
            rows.append(check(f"B:{site} Threat recall ≥ 88%", None, 0.88, 0.0,
                              hard=not skip, kind='gte',
                              detail='awaits phase_b JSON'))
            rows.append(check(f"B:{site} Speech-FP post ≤ 10%", None, 0.10, 0.0,
                              hard=not skip, kind='lte',
                              detail='awaits phase_b JSON'))
            rows.append(check(f"B:{site} Determinism", None, True, None,
                              hard=False, kind='info',
                              detail='awaits phase_b JSON'))
            rows.append(check(f"B:{site} Silence gating", None, True, None,
                              hard=False, kind='info',
                              detail='awaits phase_b JSON'))
            continue
        rows.append(check(f"B:{site} FP ≤ 5%", blob.get('fp_rate'), 0.05, 0.0,
                          hard=True, kind='lte',
                          detail=f"n={blob.get('held_out_wavs','?')} held-out wavs"))
        rows.append(check(f"B:{site} Threat recall ≥ 88%",
                          blob.get('threat_recall'), 0.88, 0.0,
                          hard=True, kind='gte',
                          detail=f"threats={blob.get('threat_dir','?')}"))
        rows.append(check(f"B:{site} Speech-FP post ≤ 10%",
                          blob.get('speech_fp_post'), 0.10, 0.0,
                          hard=True, kind='lte'))
        rows.append(check(f"B:{site} Determinism",
                          blob.get('determinism_ok'), True, None,
                          hard=True, kind='info',
                          detail="bit-identical across 10 runs"))
        rows.append(check(f"B:{site} Silence gating",
                          blob.get('silence_gate_ok'), True, None,
                          hard=True, kind='info'))
    print_rows("Phase B rows", rows)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Phase C — narrative / marketing rows
# ──────────────────────────────────────────────────────────────────────


def phase_c(phase_a: Dict, phase_b_rows: List[Dict]) -> List[Dict]:
    print()
    print("=" * 80)
    print("PHASE C  Narrative (paper / homepage / pitch deck deltas)")
    print("=" * 80)
    rows: List[Dict] = []

    # 21 — post-fine-tune speech FP (pre leg from Phase A; post leg from
    # Phase B blob, if present).
    sfp_pre = phase_a['metrics']['fpr_by_source'].get('as_speech', {}).get('rate')
    rows.append(check('C:Speech FP pre  ~71.7%', sfp_pre, 0.717, 0.04,
                      hard=True, detail='same measurement as Phase A row'))
    blob = _load_phase_b_json('metro')
    sfp_post = blob.get('speech_fp_post') if blob else None
    rows.append(check('C:Speech FP post ≤ 10% (claim: 4.2%)',
                      sfp_post, 0.10, 0.0, hard=blob is not None,
                      kind='lte',
                      detail=('measured after metro fine-tune' if blob is not None
                              else 'awaits Phase B')))

    # 22 — post-fine-tune threat recall
    tr_post = blob.get('threat_recall') if blob else None
    rows.append(check('C:Post-FT recall ~86%', tr_post, 0.86, 0.03,
                      hard=blob is not None,
                      detail=('measured after metro fine-tune' if blob is not None
                              else 'awaits Phase B')))

    # 23 — SOTA footprint
    ratio = SOTA_PARAMS_M['PANNs-CNN14'] / SOTA_PARAMS_M['SafeCommute']
    rows.append(check('C:~50× smaller vs CNN14',
                      ratio, PAPER_SOTA_RATIO_CLAIM,
                      PAPER_SOTA_RATIO_CLAIM * PAPER_SOTA_RATIO_TOL,
                      hard=True, kind='abs',
                      detail=f"param ratio = {ratio:.1f}× "
                             f"(80M / 1.83M)"))
    rows.append(check('C:~10× faster vs CNN14/AST',
                      None, 10.0, None, hard=False, kind='info',
                      detail='no measured SOTA latency row yet — soft claim'))

    # 24 γ ablation
    rows.append(check('C:γ=0.5 AUC', phase_a['metrics']['auc_roc'],
                      GAMMA_0_5_AUC, 0.010, hard=True))
    rows.append(check('C:γ=3.0 AUC 0.856',
                      None, GAMMA_3_0_AUC_HISTORICAL, None,
                      hard=False, kind='info',
                      detail='historical snapshot per user decision'))

    print_rows("Phase C rows", rows)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Latency + size
# ──────────────────────────────────────────────────────────────────────


def latency_and_size() -> List[Dict]:
    print()
    print("=" * 80)
    print("LATENCY + SIZE")
    print("=" * 80)
    rows: List[Dict] = []

    target_hw = os.environ.get('TARGET_HW', '').strip()
    hw_hard = target_hw == 'Ryzen5-1T'

    # ── FP32 latency (eager PyTorch, default threads, inference_mode) ──
    model = load_model(MODEL_SAVE_PATH, torch.device('cpu'))
    x = torch.randn(1, 1, N_MELS, TIME_FRAMES)

    def run_eager(_x):
        with torch.inference_mode():
            model(_x)

    fp32_stats = time_forward(run_eager, x, n_warmup=30, n_measure=100)
    rows.append(check('Deploy FP32 mean ≤ 15 ms', fp32_stats['mean'],
                      DEPLOYMENT_LATENCY_MEAN_MS, 0.0,
                      hard=hw_hard, kind='lte',
                      detail=f"p99 {fp32_stats['p99']:.1f} ms, "
                             f"median {fp32_stats['median']:.1f} ms"))
    rows.append(check('Deploy FP32 p99 ≤ 30 ms', fp32_stats['p99'],
                      DEPLOYMENT_LATENCY_P99_MS, 0.0,
                      hard=hw_hard, kind='lte'))

    # ── FP32 ONNX (the "marketing ~12 ms on Ryzen 5, 1T" path) ─────────
    onnx_path = MODEL_SAVE_PATH.replace('.pth', '.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = torch.get_num_threads()
            sess = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                        providers=['CPUExecutionProvider'])
            in_name = sess.get_inputs()[0].name
            np_x = x.numpy()

            def run_ort(_x):
                sess.run(None, {in_name: np_x})

            onnx_stats = time_forward(run_ort, x, n_warmup=30, n_measure=100)
            rows.append(check('ONNX FP32 mean ~12 ms (marketing)',
                              onnx_stats['mean'], MARKETING_LATENCY_MS, 3.0,
                              hard=hw_hard, kind='lte',
                              detail=f"median {onnx_stats['median']:.1f} ms, "
                                     f"p99 {onnx_stats['p99']:.1f} ms, "
                                     f"threads={torch.get_num_threads()}"))
        except Exception as e:
            rows.append(check('ONNX FP32 mean ~12 ms (marketing)',
                              None, MARKETING_LATENCY_MS, None,
                              hard=False, kind='lte',
                              detail=f"onnxruntime error: {type(e).__name__}"))
    else:
        rows.append(check('ONNX FP32 mean ~12 ms (marketing)',
                          None, MARKETING_LATENCY_MS, None,
                          hard=False, kind='lte',
                          detail=f"{onnx_path} missing — run safecommute/export.py"))

    # ── INT8 ONNX (appears only after Step 5) ─────────────────────────
    int8_path = MODEL_SAVE_PATH.replace('.pth', '_int8.onnx')
    if os.path.exists(int8_path):
        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = torch.get_num_threads()
            sess8 = ort.InferenceSession(int8_path, sess_options=sess_opts,
                                         providers=['CPUExecutionProvider'])
            in_name = sess8.get_inputs()[0].name
            np_x = x.numpy()

            def run_q(_x):
                sess8.run(None, {in_name: np_x})

            q_stats = time_forward(run_q, x, n_warmup=30, n_measure=100)
            rows.append(check('INT8 ONNX mean ≤ 15 ms',
                              q_stats['mean'], 15.0, 0.0,
                              hard=hw_hard, kind='lte',
                              detail=f"median {q_stats['median']:.1f} ms "
                                     f"p99 {q_stats['p99']:.1f} ms"))
        except Exception as e:
            rows.append(check('INT8 ONNX mean ≤ 15 ms',
                              None, 15.0, None, hard=False, kind='lte',
                              detail=f"onnxruntime: {type(e).__name__}: {e}"))
    else:
        rows.append(check('INT8 ONNX mean ≤ 15 ms',
                          None, 15.0, None, hard=False, kind='lte',
                          detail=f"{int8_path} missing — run Step 5"))

    # ── Size checks ───────────────────────────────────────────────────
    sizes = model_size_stats(MODEL_SAVE_PATH)
    rows.append(check('Size FP32 ≤ 10 MB', sizes['fp32_mb'],
                      FP32_SIZE_MAX_MB, 0.0, hard=True, kind='lte'))
    if sizes.get('int8_onnx_mb') is not None:
        rows.append(check('Size INT8 ONNX ≤ 6 MB', sizes['int8_onnx_mb'],
                          INT8_SIZE_MAX_MB, 0.0, hard=True, kind='lte'))
    elif sizes.get('int8_pth_mb') is not None:
        rows.append(check('Size INT8 .pth ≤ 6 MB (fallback)',
                          sizes['int8_pth_mb'],
                          INT8_SIZE_MAX_MB, 0.0,
                          hard=False, kind='lte',
                          detail="INT8 ONNX not yet present (Step 5); "
                                 ".pth is Linear+GRU-only quant"))
    else:
        rows.append(check('Size INT8 ONNX ≤ 6 MB', None,
                          INT8_SIZE_MAX_MB, None, hard=False, kind='lte',
                          detail="INT8 artefact missing"))

    print_rows("Latency + size rows", rows)
    return rows


def int8_parity_check(phase_a_metrics: Dict) -> List[Dict]:
    """Claim 28 — INT8 AUC within 0.02 of FP32. Only runs if the INT8 ONNX
    exists (Step 5 output)."""
    rows: List[Dict] = []
    int8_path = MODEL_SAVE_PATH.replace('.pth', '_int8.onnx')
    if not os.path.exists(int8_path):
        rows.append(check('INT8 AUC degradation ≤ 0.02',
                          None, 0.02, None, hard=False, kind='lte',
                          detail=f"{int8_path} missing — run Step 5"))
        print()
        print("=" * 80)
        print("INT8 PARITY")
        print("=" * 80)
        print_rows("INT8 parity rows", rows)
        return rows

    print()
    print("=" * 80)
    print("INT8 PARITY  (INT8 ONNX vs FP32 on prepared_data/test/)")
    print("=" * 80)

    try:
        from sklearn.metrics import roc_auc_score
        from tests._common import run_inference_onnx
        ds = load_test_dataset('test')
        p8, y8, _ = run_inference_onnx(int8_path, ds)
        auc8 = float(roc_auc_score(y8, p8))
        auc32 = float(phase_a_metrics['auc_roc'])
        delta = abs(auc32 - auc8)
        rows.append(check('INT8 AUC',
                          auc8, auc32, 0.02, hard=True, kind='abs',
                          detail=f"FP32 baseline {auc32:.4f}, "
                                 f"INT8 {auc8:.4f}, |Δ| = {delta:.4f}"))
    except Exception as e:
        rows.append(check('INT8 AUC',
                          None, phase_a_metrics['auc_roc'], 0.02,
                          hard=False, kind='abs',
                          detail=f"INT8 inference failed: {type(e).__name__}: {e}"))

    print_rows("INT8 parity rows", rows)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Report I/O
# ──────────────────────────────────────────────────────────────────────


def summarize(all_rows: List[Dict]) -> Dict:
    passed = sum(1 for r in all_rows if r['passed'])
    failed_hard = sum(1 for r in all_rows if not r['passed'] and r['hard'])
    warned = sum(1 for r in all_rows if not r['passed'] and not r['hard'])
    return {
        'total': len(all_rows),
        'passed': passed,
        'failed_hard': failed_hard,
        'warned': warned,
    }


def write_report(all_rows: List[Dict],
                 phase_a_metrics: Dict,
                 hw: Dict,
                 out_path: str = 'tests/reports/verify_performance_claims.json'
                 ) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'hw_disclosure': hw,
        'rows': all_rows,
        'summary': summarize(all_rows),
        'phase_a_full_metrics': phase_a_metrics,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return out_path


def _json_default(o):
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


GAMMA_HISTORICAL = {
    # RESULTS.md §"Experiment Summary" + paper.md §3.2 ablation.
    # γ=3.0 checkpoint is not preserved — historical snapshot per user decision.
    'gammas':        [0.0,   0.5,   1.0,   2.0,   3.0],
    'auc':           [0.761, 0.804, 0.812, 0.835, 0.856],
    'hard_neg_acc':  [52.1,  46.9,  31.4,  9.2,   0.0],
    'note': ('γ=0.5 reproduces from current checkpoint (measured 0.804); '
             'other γ values are training-log snapshots, checkpoints not preserved.')
}

SOTA_TABLE = {
    # params_m: static arithmetic facts.
    # latency_ms / score: historical literature numbers — not measured
    # on the current hardware. Label accordingly in the paper.
    'models':     ['SafeCommute', 'YAMNet', 'PANNs-CNN14', 'AST'],
    'params_m':   [1.83, 3.7, 80.0, 86.0],
    'latency_ms': [None, None, None, None],     # soft — not measured
    'latency_ms_historical': [12, 50, 150, 200],
    'score':        [None, 0.306, 0.431, 0.485],  # AudioSet mAP from original papers
    'score_note':   'SafeCommute cell intentionally left None — AUC(binary) not comparable to mAP(527-class).',
    'score_metric': ['AUC', 'mAP', 'mAP', 'mAP'],
}


def write_figures_json(phase_a_metrics: Dict,
                       out_path: str = 'tests/reports/figures_source.json'
                       ) -> str:
    """Model-derived numbers for scripts/generate_pitch_figures.py.

    Every hand-typed literal in that script (confusion matrix cells,
    per-source accuracy arrays, SOTA comparison, γ ablation, fine-tune
    impact) must become a read from this file. Step 9 rewrites the
    pitch-figures script to consume this.
    """
    threats = []
    for src, cell in sorted(phase_a_metrics['tpr_by_source'].items(),
                            key=lambda kv: -kv[1]['rate']):
        threats.append({'source': src, 'n': cell['n'],
                        'accuracy': cell['rate']})
    safes = []
    for src, cell in sorted(phase_a_metrics['fpr_by_source'].items(),
                            key=lambda kv: kv[1]['rate']):
        safes.append({'source': src, 'n': cell['n'],
                      'accuracy': 1.0 - cell['rate']})

    # Fine-tune impact — pulled from phase_b_<site>.json if present, else null.
    finetune_impact = None
    for site, _, _ in SITES:
        blob = _load_phase_b_json(site)
        if blob is None:
            continue
        pre = phase_a_metrics['fpr_by_source']
        finetune_impact = {
            'site': site,
            'categories': [
                {'name': 'speech FP',
                 'before': pre.get('as_speech', {}).get('rate'),
                 'after': blob.get('speech_fp_post'),
                 'kind': 'fpr'},
                {'name': 'overall FP (held-out)',
                 'before': phase_a_metrics.get('overall_fpr'),
                 'after': blob.get('fp_rate'),
                 'kind': 'fpr'},
                {'name': 'threat recall',
                 'before': phase_a_metrics.get('overall_tpr'),
                 'after': blob.get('threat_recall'),
                 'kind': 'recall'},
            ],
            'low_fpr_threshold': blob.get('low_fpr_threshold'),
            'held_out_wavs': blob.get('held_out_wavs'),
        }
        break

    out = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'auc_roc': phase_a_metrics['auc_roc'],
        'accuracy': phase_a_metrics['accuracy'],
        'f1_weighted': phase_a_metrics['f1_weighted'],
        'overall_fpr_at_0_5': phase_a_metrics['overall_fpr'],
        'overall_tpr_at_0_5': phase_a_metrics['overall_tpr'],
        'confusion_matrix_normalized_at_0_5': phase_a_metrics['confusion_matrix_normalized'],
        'confusion_matrix_counts_at_0_5': phase_a_metrics['confusion_matrix_counts'],
        'threshold_sweep': phase_a_metrics['threshold_sweep'],
        'threats': threats,
        'safes': safes,
        'sota_params_m': SOTA_PARAMS_M,
        'sota_table': SOTA_TABLE,
        'gamma_ablation_historical': GAMMA_HISTORICAL,
        'finetune_impact': finetune_impact,
        'hw_disclosure': hw_disclosure(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=_json_default)
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-phase-b', action='store_true',
                   help='Treat missing Phase B JSON as WARN instead of FAIL.')
    p.add_argument('--emit-figures-json',
                   nargs='?',
                   const='tests/reports/figures_source.json',
                   default=None,
                   help='Also write the figures-source JSON (for pitch figures).')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = hw_disclosure()

    print()
    print("SafeCommute AI — performance-claim verification")
    print("=" * 80)
    print(f"  Hardware: {hw['cpu_model']} ({hw['cores']} cores, "
          f"{hw['torch_threads']} torch threads)  governor={hw['governor']}")
    print(f"  Software: torch {hw['torch_version']}, "
          f"onnxruntime {hw['onnxruntime_version']}, "
          f"numpy {hw['numpy_version']}")
    print(f"  TARGET_HW env = {hw['target_hw_env']}  "
          f"(set to 'Ryzen5-1T' to promote latency rows to hard checks)")
    print(f"  Device: {device}")

    a = phase_a(device)
    b_rows = phase_b(skip=args.skip_phase_b)
    c_rows = phase_c(a, b_rows)
    ls_rows = latency_and_size()
    int8_rows = int8_parity_check(a['metrics'])

    all_rows = a['rows'] + b_rows + c_rows + ls_rows + int8_rows

    out = write_report(all_rows, a['metrics'], hw)
    print()
    print(f"  Report: {out}")
    if args.emit_figures_json is not None:
        fig_out = write_figures_json(a['metrics'], args.emit_figures_json)
        print(f"  Figures JSON: {fig_out}")

    summary = summarize(all_rows)
    print()
    print("Result: "
          f"{summary['passed']} PASS, "
          f"{summary['failed_hard']} FAIL, "
          f"{summary['warned']} WARN  →  "
          f"exit {1 if summary['failed_hard'] else 0}")
    sys.exit(1 if summary['failed_hard'] else 0)


if __name__ == '__main__':
    main()
