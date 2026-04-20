# Deploy the SafeCommute web + dashboard

This guide ships the marketing site, the demo download, and the paid fine-tune dashboard in one Vercel deployment backed by Supabase (DB + Auth + Storage) and Stripe (payments).

All commands assume **`cwd = web/`** (i.e. `cd web` from the repo root first).

---

## 0. What you'll end up with

- `https://<your-project>.vercel.app` — public landing page (the Bauhaus one).
- `https://<your-project>.vercel.app/demo/safecommute-v2-demo.zip` — the 6.5 MB demo bundle (model + `short.md`), served as a static file.
- `https://<your-project>.vercel.app/dashboard` — authenticated dashboard. Users sign in via email magic link, buy credits (€100 subscription or €23 per-run), upload audio clips, trigger queued fine-tune jobs, see job status, download the calibrated model.

All dashboard data (users, sites, clips, jobs, payments) lives in Supabase. No data is stored on Vercel.

---

## 1. Build the demo bundle (runs automatically on every build)

```bash
npm install
npm run bundle:demo   # or just npm run build — prebuild hook does this
```

`scripts/build-demo-bundle.mjs` zips `../models/safecommute_v2.pth` + `public/demo/short.md` into `public/demo/safecommute-v2-demo.zip`. Check the zip into git so Vercel builds don't need the `.pth` on the build host:

```bash
git add public/demo/safecommute-v2-demo.zip public/demo/short.md
git commit -m "ship demo bundle"
```

---

## 2. Supabase — 5 minutes

1. Create a project at <https://supabase.com/dashboard>. Pick the EU region if your users are European.
2. **SQL editor → New query**, paste `supabase/migrations/0001_init.sql` (relative to `web/`), run. This creates the tables, RLS policies, the `audio-uploads` Storage bucket, and a trigger that autocreates `profiles` + `entitlements` rows when a user signs up.
3. **Authentication → Providers**: enable Email (magic link is on by default). Optionally enable Google.
4. **Authentication → URL configuration**: add `https://<your-vercel-domain>/api/auth/callback` to the allowed redirect URLs (and `http://localhost:3000/api/auth/callback` for dev).
5. **Project settings → API**: copy `Project URL`, `anon public`, and `service_role` keys — you'll paste them into Vercel in step 4.

---

## 3. Stripe — 5 minutes

1. Create an account at <https://dashboard.stripe.com> and switch to **test mode** first.
2. **Products → Add product**:
   - "SafeCommute Subscription — one site" · one-time price €100 EUR · copy the `price_…` id into `STRIPE_PRICE_SUBSCRIPTION`.
   - "SafeCommute Per-run credit" · one-time price €23 EUR · copy the `price_…` id into `STRIPE_PRICE_PER_RUN`.
   - (Both are one-time payments — the "subscription" label is a product name, not a Stripe recurring subscription. This matches the current UX; if you later want real recurring, switch the product to recurring and flip the Checkout session `mode` to `"subscription"`.)
3. **Developers → API keys**: copy the test secret key into `STRIPE_SECRET_KEY`, publishable key into `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`.
4. **Developers → Webhooks → Add endpoint**: URL `https://<your-vercel-domain>/api/stripe/webhook`, select event `checkout.session.completed`. Copy the signing secret into `STRIPE_WEBHOOK_SECRET`.

---

## 4. Vercel — 2 minutes

```bash
npx vercel login         # one-time
npx vercel link          # creates the project, links this directory
```

Set environment variables (do this for both `Preview` and `Production`):

```bash
npx vercel env add NEXT_PUBLIC_SUPABASE_URL
npx vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY
npx vercel env add SUPABASE_SERVICE_ROLE_KEY
npx vercel env add STRIPE_SECRET_KEY
npx vercel env add STRIPE_WEBHOOK_SECRET
npx vercel env add STRIPE_PRICE_SUBSCRIPTION
npx vercel env add STRIPE_PRICE_PER_RUN
npx vercel env add NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
npx vercel env add NEXT_PUBLIC_SITE_URL      # e.g. https://safecommute.vercel.app
```

Deploy:

```bash
npx vercel --prod
```

The CLI prints a shareable URL — that's the live site.

Finally: go back to Supabase and add the Vercel domain to the auth redirect URL list, and go to Stripe and confirm the webhook endpoint resolves. Redeploy after setting `NEXT_PUBLIC_SITE_URL` to the final domain.

---

## 5. What ships without external services

Even before you wire Supabase + Stripe, the deployment works partially:

- Landing page: **works**.
- Demo download: **works** (static file).
- `/dashboard`, `/sign-in`: **show the "Dashboard offline" Bauhaus error card** with a download-demo CTA — the user sees a polished fallback instead of a 500.

This lets you ship the public URL first, then layer the paid flow on.

---

## 6. Fine-tune worker (not shipped in this tick)

The dashboard queues jobs — it doesn't run them. To wire a real worker:

1. Deploy a Python container on Railway / Fly / a cheap GPU VM that:
   - polls `finetune_jobs where status='queued'` with the service-role key,
   - downloads the corresponding `audio_clips` from the `audio-uploads` bucket,
   - runs `safecommute/pipeline/finetune.py --environment <site.name> --ambient-dir <tempdir> --freeze-cnn`,
   - uploads the produced `{environment}_model.pth` back to a new Storage bucket `fine-tuned-models`,
   - updates `finetune_jobs` row to `status='succeeded'` with `model_path` set.
2. Add a `/api/finetune/[id]/download` route that issues a signed URL from `fine-tuned-models`.

Until then, paying users get a "Queued" state and can see their progress UI — which is what the plan approved for this tick.

---

## 7. Sanity checklist after deploy

- [ ] Open the Vercel URL in an incognito window. Landing page renders with animations.
- [ ] Click "Download demo ↓" in the hero. A 6.5 MB zip arrives. Unzipped contents: `safecommute_v2.pth` (7.3 MB) and `short.md`.
- [ ] Click "Open dashboard →". You land on `/sign-in`. Enter your email, get a magic link, click it, land on `/dashboard`.
- [ ] `/dashboard/billing` → "Pay €23 →" → Stripe Checkout (use test card `4242 4242 4242 4242`, any future date, any CVC) → success → back to billing with "Payment received" banner and `per_run_credits = 1`.
- [ ] Create a site, drop 3 WAV files into the drop zone, click "Run fine-tune" → the credit drops to 0 and a "Queued" job card appears with the rotating ring + waveform pulse.
- [ ] In Supabase dashboard, confirm rows exist in `sites`, `audio_clips`, `finetune_jobs`, `payments`, `entitlements`, and that the audio files are in Storage → `audio-uploads/<your-uid>/<site-id>/`.
