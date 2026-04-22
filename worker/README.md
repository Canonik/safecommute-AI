# SafeCommute fine-tune worker

Python background worker that consumes queued jobs from the Supabase
`finetune_jobs` table, runs the tweak-3 per-site fine-tune + INT8 ONNX export
+ deployment gate, and publishes the artefacts back to Supabase Storage. This
is the piece that lets a paying customer actually complete a flow:

```
upload clips ─► pay ─► trigger (POST /api/finetune/trigger) ─► [WORKER] ─► download
                                                                  ↑
                                                        this directory
```

Prior to 2026-04-22 the worker did not exist — the trigger route inserted
`status='queued'` rows that nobody consumed. CLAUDE.md §"Website backend
state" flagged this as the #1 shipping blocker.

## Layout

- `worker/main.py` — loop: claim → run → mark done → sleep.
- `worker/job.py` — per-job processor: download clips, split 80/20, fine-tune,
  quantize, deployment report, upload artefacts, delete source clips.
- `worker/export.py` — inline FP32 → INT8 ONNX helper (per-job paths) that
  reuses the low-level functions in `safecommute/export_quantized.py`.
- `worker/supabase_client.py` — thin REST + Storage wrapper for the handful
  of Supabase calls the worker makes (avoids pulling in supabase-py).

## Setup

1. Run both SQL migrations against your Supabase project:

   ```bash
   # If you use the Supabase CLI:
   supabase db push

   # Or paste the contents of each into the SQL editor, in order:
   #   web/supabase/migrations/0001_init.sql
   #   web/supabase/migrations/0002_models_deliverable_bucket.sql
   ```

2. Copy the env template and fill in real values:

   ```bash
   cp worker/.env.example worker/.env
   $EDITOR worker/.env
   ```

   The service-role key (`SUPABASE_SERVICE_ROLE_KEY`) bypasses RLS — never
   commit this file or ship it into a browser bundle.

3. Ensure `requirements.txt` is installed into the venv used by the systemd
   unit:

   ```bash
   source venv/bin/activate.fish
   pip install -r requirements.txt
   ```

## Running locally

```bash
set -a; source worker/.env; set +a
PYTHONPATH=. python -m worker.main
```

The worker polls every `WORKER_POLL_INTERVAL_S` seconds (default 15).
`LOG_LEVEL=DEBUG` tips detailed per-step logs.

## Running as a systemd user unit (self-hosted recipe)

The finalization plan calls for self-hosted v1 on the Ryzen 7 7435HS dev box
— zero cloud cost, fine until ~5 paying users.

```bash
# Copy the unit to the user-systemd dir
install -Dm0644 systemd/safecommute-worker.service \
  ~/.config/systemd/user/safecommute-worker.service

# Reload + enable + start
systemctl --user daemon-reload
systemctl --user enable --now safecommute-worker.service

# Tail
journalctl --user -u safecommute-worker.service -f

# Linger (so the worker keeps running across SSH logouts)
sudo loginctl enable-linger $(whoami)
```

The unit's `WorkingDirectory` + `ExecStart` assume the repo lives at
`$HOME/github/safecommute-AI/` and the venv is at
`$HOME/github/safecommute-AI/venv/`. Adjust if your layout differs.

## How a job is processed

For each claimed row (status flipped `queued → running` atomically):

1. **Download clips** from `audio-uploads/{owner}/{site_id}/` into a per-job
   temp dir.
2. **Deterministic 80/20 split** on the clips' SHA256(salt + basename)
   bucket. 80% feed `safecommute/pipeline/finetune.py --ambient-dir …`; 20%
   feed `--calibration-ambient-dir …` so the emitted `low_fpr_site` threshold
   is calibrated on held-out customer ambient, not the universal test set.
3. **Fine-tune** with the tweak-3 recipe (`--keep-safe-ratio 0.1 --epochs 20
   --freeze-cnn`) — the best architecture-preserving recipe from paper.md
   §1.4, the one that combined with `--majority-k 2` cleared the FP gate on
   metro.
4. **Static INT8 ONNX export** via `worker/export.py` (reuses
   `safecommute.export_quantized` helpers but with per-job paths so the base
   `models/safecommute_v2_int8.onnx` artefact is never touched).
5. **Deployment report** — `safecommute/pipeline/test_deployment.py
   --majority-k 2 --ambient-dir <customer cal> --threat-dir
   raw_data/youtube_screams`. Output stashed for the customer alongside the
   model.
6. **Upload** three files to
   `models-deliverable/{owner}/{job_id}/{model.onnx, thresholds.json,
   deployment_report.json}`. Signed-URL downloads routed via
   `/api/finetune/[id]/download?file=model|thresholds|report`.
7. **Mark succeeded** — updates `status`, `model_path` (the bucket prefix),
   `thresholds` (JSONB), `completed_at`. The dashboard job card watches
   `status='succeeded' && model_path` to render the 3 download buttons.
8. **Delete raw clips** from `audio-uploads` — the privacy fix. Raw audio is
   ephemeral; only the non-invertible PCEN-derived INT8 weights persist.

Anything that throws in steps 1–8 turns into `status='failed'`, a short
error string in `finetune_jobs.error` (what the customer sees), and the full
traceback in `worker_logs` (what the operator greps).

## Env var overrides for the recipe

`FT_KEEP_SAFE_RATIO`, `FT_EPOCHS`, `FT_LR`, `MAJORITY_K` all have sane
defaults (see `worker/.env.example`). Override only if you're intentionally
deviating from the paper's recipe — e.g. a faster smoke-test run.
