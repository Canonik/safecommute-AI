# Deployment Next Steps

Concrete runbook to take the SafeCommute paid fine-tuning product from
"all code landed, nothing deployed" (the 2026-04-22 state) to
"a paying customer completes upload → pay → download on production".

This document is **ordered by blocker priority** — do items in order unless
noted. Each step is either a ≤ 30-min operator task (commands provided) or
flagged as "needs user-side action" (field recording, Stripe account setup,
etc.).

Context: see [paper.md §0 TL;DR](paper.md) for the research state,
[VALIDATE_AND_IMPROVE.md §14](VALIDATE_AND_IMPROVE.md) for what landed this
session, and [worker/README.md](worker/README.md) for the worker's
per-job protocol.

---

## 0. State as of 2026-04-22

| Layer | Status |
|---|---|
| Marketing site (`web/`) | deployed at <https://safecommute-ai.vercel.app> |
| Supabase migration 0001 (profiles / entitlements / sites / clips / jobs / payments / `audio-uploads` bucket) | deployed, working |
| Supabase migration **0002** (new `models-deliverable` bucket + `worker_logs` table) | **not yet applied** to the production project |
| Trigger route (`POST /api/finetune/trigger`) | deployed, queues jobs as `status='queued'` |
| Fine-tune worker | **code complete, not running anywhere yet** |
| Download route (`GET /api/finetune/[id]/download`) | **code merged, needs redeploy** |
| Dashboard download buttons | **code merged, needs redeploy** |
| Privacy section copy (ephemeral-bucket reality) | **code merged, needs redeploy** |
| Stripe in test mode | working (€23 per-run credit + €100 site unlock) |
| Stripe live mode | **never activated** |
| Base model artefacts in git (pth + INT8 ONNX + gamma3) | present under `models/` |
| Field-recorded sites for Phase B | **n=1 (metro YouTube clips)** — need ≥ 2 more |

**Hard blocker chain for first paying customer**: 0002 migration → worker
process running → Vercel redeploy → end-to-end smoke test.

---

## 1. Apply Supabase migration 0002 (5 min)

The worker writes to a new `models-deliverable` bucket and emits tracebacks
to a new `worker_logs` table. Both are created by the migration.

```bash
# If you use the Supabase CLI (recommended — idempotent, traceable):
cd web
supabase db push

# Or paste the SQL by hand:
# 1. Open https://app.supabase.com/project/<id>/sql/new
# 2. Paste the full contents of web/supabase/migrations/0002_models_deliverable_bucket.sql
# 3. Click "Run". Expect "Success. No rows returned."
```

Verify: in the Supabase dashboard, **Storage** should show both
`audio-uploads` and `models-deliverable` buckets. **Table editor** should
list a new `worker_logs` table under the `public` schema.

---

## 2. Provision the worker host (Ryzen dev box, systemd user unit)

Per the finalization plan, v1 runs self-hosted. Cloud migration is
deferred until the product has ≥ 5 paying users.

### 2.1 Fill worker/.env

```bash
cd ~/github/safecommute-AI
cp worker/.env.example worker/.env
$EDITOR worker/.env
```

Required keys (from Supabase **Project Settings → API**):

- `SUPABASE_URL=https://<project>.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...`  *(bypasses RLS — never commit)*

Defaults for the rest are sane:
`WORKER_POLL_INTERVAL_S=15`, `WORKER_CONCURRENCY=1`, `FT_KEEP_SAFE_RATIO=0.1`,
`FT_EPOCHS=20`, `MAJORITY_K=2`.

### 2.2 Install the systemd user unit

```bash
# Make sure deps are in the venv the unit uses
source venv/bin/activate.fish
pip install -r requirements.txt

# Copy the unit file
install -Dm0644 systemd/safecommute-worker.service \
  ~/.config/systemd/user/safecommute-worker.service

systemctl --user daemon-reload
systemctl --user enable --now safecommute-worker.service

# Keep it running after SSH logout
sudo loginctl enable-linger $(whoami)
```

### 2.3 Confirm it's alive

```bash
systemctl --user status safecommute-worker.service
# → Active: active (running)

journalctl --user -u safecommute-worker.service -f
# → Expect: "safecommute worker starting (poll=15s)"
# → Then every 15s: nothing (no queued jobs yet — correct)
```

Troubleshooting: if the service `failed`, check
`journalctl --user -u safecommute-worker.service -n 100`. The most common
cause is a wrong `SUPABASE_SERVICE_ROLE_KEY` — the worker will log
`supabase rest PATCH /finetune_jobs -> 401` on every poll.

---

## 3. Redeploy the Vercel project (2 min)

The download route, dashboard buttons, and privacy section copy shipped in
commit `bcc18d9` need a production deploy.

```bash
cd web
vercel --prod
```

If Vercel is wired to auto-deploy from `main`, a `git push origin main`
will trigger the deploy. Watch <https://vercel.com/canonik/safecommute-ai/deployments>
for the green "Ready" badge.

Verify:

- <https://safecommute-ai.vercel.app/#privacy> — copy should say
  "At inference, no audio leaves the device. For fine-tuning, audio is
  ephemeral." (the two-part framing from the new
  `privacy-section.tsx`).
- Load a succeeded job in the dashboard — should show three buttons
  (`Model .onnx ↓`, `Thresholds ↓`, `Deployment report ↓`).

---

## 4. Smoke-test end-to-end in Stripe test mode (30 min)

The first thing a paying customer does should be something you've done
yourself on a throwaway account, at least once.

1. Open <https://safecommute-ai.vercel.app/dashboard> in a private window.
2. Sign up with a throwaway email (magic link).
3. **Create a site** (`+ Add site` button). Pick environment = `retail`.
4. **Upload 3 wav clips** from a public-domain ambient corpus (e.g. pick
   three `raw_data/youtube_metro_quarantine/*.wav` and upload via the
   dashboard's file-picker). The uploader should show all three as
   "uploaded".
5. **Pay for a per-run credit** (€3 per-run button) using the Stripe test
   card `4242 4242 4242 4242`, any future expiry, any CVC.
6. Back on the site page, click **"Run fine-tune"**. The dashboard should
   flip to "queued", then "running" within 15s. Watch the worker:

   ```bash
   journalctl --user -u safecommute-worker.service -f
   ```

   Expect: "claimed job <uuid>", "processing job …", "running finetune.py …",
   "running export …", "uploaded artefacts to models-deliverable/…/",
   "deleted 3 source clips from audio-uploads".

7. After ~10–20 minutes (20-epoch fine-tune on CPU), the dashboard flips
   to "succeeded" and shows the three download buttons.
8. Click each button:
   - `Model .onnx ↓` should download a ~3.7 MB `.onnx` file.
   - `Thresholds ↓` should download a tiny `thresholds.json` containing
     `{youden, f1_optimal, low_fpr, low_fpr_site, ...}`.
   - `Deployment report ↓` should download the `test_deployment.py`
     stdout as JSON.
9. Load the downloaded `.onnx` in a throwaway Python script and confirm
   it returns a probability:

   ```python
   import onnxruntime as ort, numpy as np
   s = ort.InferenceSession("model.onnx")
   x = np.random.randn(1, 1, 64, 188).astype("float32")
   print(s.run(None, {s.get_inputs()[0].name: x})[0])
   ```

10. Confirm the source clips are gone:
    Supabase dashboard → Storage → `audio-uploads` → `<user.id>/<site.id>/`
    should be empty.

If any of steps 6–10 fails, `worker_logs` will have the traceback; fix
before any real customer hits the flow.

---

## 5. Switch Stripe to live mode (user decision + 15 min)

Only flip this after step 4 passes cleanly. Stripe's live mode
key-rotation is irreversible in the sense that live payments are real.

1. <https://dashboard.stripe.com>, toggle **Test mode OFF**.
2. Re-create both products (€23 per-run, €100 site unlock) as one-time
   prices in EUR. Copy the new `price_1...` ids.
3. Vercel → Project Settings → Environment Variables. Overwrite
   `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`,
   `STRIPE_PRICE_SUBSCRIPTION`, `STRIPE_PRICE_PER_RUN` with the live
   values. Redeploy (`vercel --prod`).
4. Stripe → **Developers → Webhooks → + Add endpoint**:
   `https://safecommute-ai.vercel.app/api/stripe/webhook`, subscribe to
   `checkout.session.completed`. Copy the signing secret into
   `STRIPE_WEBHOOK_SECRET` (it changes per environment).
5. Smoke-test again with a real €3 charge on your own card; refund
   yourself from the Stripe dashboard after verifying the
   `entitlements` row incremented.

---

## 6. Honest-claims marketing pass (Phase 3, 2–4 hours)

Before the first paying customer, the marketing copy should match the
measured numbers the paper now reports. Per the finalization plan §3
("Honest-claims hygiene"), three specific edits:

- **[web/components/hero.tsx](web/components/hero.tsx)** — "12 ms" disc:
  either replace with "2.8 ms INT8 on Ryzen 7 @ 8T" or keep the 12 ms
  disc with a footnote linking to the measured table.
- **[web/components/how-it-works.tsx](web/components/how-it-works.tsx)**
  step 04 — "Speech FP 72% → <5%". Replace with the honest measured
  delta: "Speech FP 72% → 16.9% on universal speech subset after
  metro fine-tune; FP on held-out site ambient drops to 0.0% at
  majority-k=2".
- **[web/components/pricing.tsx](web/components/pricing.tsx)** — rename
  the "Subscription" tier label to **"Site unlock — one-time,
  unlimited runs per site forever"** (matches the CLAUDE.md / DEPLOY_WEB.md
  product design).

All three are copy-only; no new components. `tsc --noEmit` + `next lint`
should stay clean.

---

## 7. Field-record ≥ 2 more sites (user-side, 1–2 hours per site)

The paper's only remaining hard blocker for workshop submission is n=1.
Infrastructure is in place — the workflow per site is one command:

```bash
# 1. Record 30–60 min of ambient in the target environment (phone mic is fine).
#    Place raw .wav files in raw_data/<site>/ and raw_data/<site>_heldout/.
#    Even split is fine; the worker handles 80/20 deterministically on its own.

# 2. Fine-tune
PYTHONPATH=. python safecommute/pipeline/finetune.py \
  --environment <site> \
  --ambient-dir raw_data/<site>/ \
  --freeze-cnn --keep-safe-ratio 0.1 --epochs 20 \
  --calibration-ambient-dir raw_data/<site>_heldout/ \
  --calibration-majority-k 2

# 3. Evaluate under the levers
echo '[{"name":"<site>_default","model_path":"models/<site>_model.pth","thresholds_file":"models/<site>_thresholds.json"}]' \
  > tests/reports/<site>_checkpoints.json

PYTHONPATH=. python tests/eval_metro_with_levers.py \
  --site <site> \
  --held-out-dir raw_data/<site>_heldout \
  --checkpoints-file tests/reports/<site>_checkpoints.json \
  --primary-checkpoint <site>_default

# 4. Refresh the verifier + paper.md numbers
PYTHONPATH=. python tests/pick_best_phase_b.py  # expects phase_b_metro.json; extend for other sites
PYTHONPATH=. python tests/verify_performance_claims.py --emit-figures-json
PYTHONPATH=. python tests/finalize_release.py
```

Target 2 additional sites. Good candidates: a café / bar (speech +
music-dominant ambient), a train platform or bus depot (mechanical noise),
an apartment / office (HVAC + conversation). Variety beats more samples
from the same environment.

When `tests/reports/phase_b_<site>.json` exists for all three sites and
the verifier reports `FP ≤ 5 %` on each, the paper crosses the n=3 bar
and §5 Limitations can drop the "n=1 site" line.

---

## 8. After the first customer — non-blocking polish

Everything below is valuable but not a launch blocker:

- **SDK + integration guide** (`docs/sdk/integration.md`) so customers have
  a copy-paste Python / Node / microcontroller snippet for loading the
  INT8 ONNX into a production edge box. See [paper.md §3.3](paper.md) for
  the input contract (16 kHz mono, 3 s window, PCEN, `(1,1,64,188)`
  tensor).
- **Stripe Customer Portal** so customers can update billing / download
  past receipts without emailing you.
- **Compliance docs** (`docs/security.md`, `docs/dpa.md`,
  `docs/retention.md`) — required before selling into transit / schools /
  elder-care / industrial.
- **Rate limits** on the trigger route (≥ 5 queued / 1 running per user)
  so a single customer can't pin the worker.
- **Case studies** — once the n=3 sites exist, write up each site's
  before/after numbers as a marketing artefact.
- **Raspberry Pi latency measurement** — the paper's "RPi 4+ ARM-ready"
  line is aspirational until a Pi run lands in
  `tests/reports/baselines_rpi.json`.
- **Client-side WASM PCEN** (§10.3 option a) — lets us upgrade the
  privacy claim from "ephemeral + PCEN-after-upload" to "never leaves
  device as raw audio". 1–2 weeks of work; defer until a customer asks.

---

## Quick reference

| Task | Command |
|---|---|
| Apply migration 0002 | `supabase db push` (from `web/`) or paste `web/supabase/migrations/0002_models_deliverable_bucket.sql` into SQL editor |
| Start worker | `systemctl --user enable --now safecommute-worker.service` |
| Stop worker | `systemctl --user stop safecommute-worker.service` |
| Tail worker | `journalctl --user -u safecommute-worker.service -f` |
| Redeploy web | `cd web && vercel --prod` |
| Smoke-test flow | See §4 above |
| Add a new site | See §7 above |
| Re-verify everything | `PYTHONPATH=. python tests/verify_performance_claims.py --emit-figures-json && PYTHONPATH=. python tests/finalize_release.py` |

---

## Sign-off gates

Before claiming "the product is live":

- [ ] Migration 0002 applied (verify in Supabase dashboard)
- [ ] Worker systemd unit active + lingered
- [ ] Vercel production URL returns updated privacy-section copy
- [ ] End-to-end smoke test with Stripe test card passes (step 4)
- [ ] Stripe flipped to live mode + webhook re-subscribed (step 5)
- [ ] `tsc --noEmit` exits 0 on `web/`
- [ ] `PYTHONPATH=. python tests/verify_performance_claims.py` exits as
      expected (currently 1 for the 6 honest FAILs documented in
      [paper.md §0](paper.md))
- [ ] First real customer's job has status `succeeded` in
      `finetune_jobs` with a non-null `model_path`
- [ ] Their clips are gone from `audio-uploads` (privacy-fix verified on
      a real row)
