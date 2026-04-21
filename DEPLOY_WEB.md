# Deploy the SafeCommute web + dashboard

Ships the marketing site, the demo download, and the paid fine-tune dashboard in one Vercel deployment backed by Supabase (DB + Auth + Storage) and Stripe (payments).

**Current production URL:** <https://safecommute-ai.vercel.app>

All commands assume **`cwd = web/`** (i.e. `cd web` from the repo root first).

> **If you're returning to this project and things already exist** (Supabase project, Vercel project linked, env vars set), skip to **[Section 7 — Returning to an existing deploy](#7-returning-to-an-existing-deploy)** first to orient yourself.

---

## 0. What you'll end up with

- `https://safecommute-ai.vercel.app` — public landing page (Bauhaus).
- `https://safecommute-ai.vercel.app/demo/safecommute-v2-demo.zip` — 6.5 MB demo bundle (model + `short.md`).
- `https://safecommute-ai.vercel.app/dashboard` — authenticated dashboard. Magic-link sign-in, one-time €100 site unlock (unlimited runs on one site) OR one-time €23 per-run credit via Stripe Checkout, audio clip uploads to Supabase Storage, queued fine-tune jobs.

All dashboard data lives in Supabase — nothing user-generated is stored on Vercel.

---

## 1. Build the demo bundle

Runs automatically on `npm run build` (via `prebuild` hook), but you can force a rebuild:

```bash
npm install
npm run bundle:demo
```

`scripts/build-demo-bundle.mjs` zips `../models/safecommute_v2.pth` + `public/demo/short.md` into `public/demo/safecommute-v2-demo.zip`. Check the zip into git so Vercel builds don't need the `.pth` on the build host:

```bash
git add public/demo/safecommute-v2-demo.zip public/demo/short.md
git commit -m "ship demo bundle"
```

Rebuild the zip whenever either `safecommute_v2.pth` or `short.md` changes (the script is idempotent — it skips if inputs are unchanged).

---

## 2. Supabase — 5 minutes

1. Create a project at <https://supabase.com/dashboard>. Pick the EU region if your users are European. Wait ~90 s for provisioning.
2. **SQL editor → New query**, paste the entire contents of `supabase/migrations/0001_init.sql` (relative to `web/`), click **Run**. You should see `Success. No rows returned.`
   - Copy to clipboard from terminal: `wl-copy < supabase/migrations/0001_init.sql` (Wayland) or `xclip -sel clip < supabase/migrations/0001_init.sql` (X11 — install xclip first).
   - Creates: `profiles`, `sites`, `audio_clips`, `finetune_jobs`, `payments`, `entitlements` tables, the private `audio-uploads` Storage bucket, RLS policies scoped to `auth.uid()`, and the `on_auth_user_created` trigger that autocreates `profiles` + `entitlements` rows on signup.
3. **Authentication → Providers**: Email is enabled by default (magic link). Optionally enable Google.
4. **Authentication → URL Configuration → Redirect URLs**: add both
   - `https://safecommute-ai.vercel.app/api/auth/callback`
   - `http://localhost:3000/api/auth/callback`
5. **Project Settings → API**: copy three values to a scratch file — you'll paste them into Vercel in section 4:

   | Supabase label | Vercel env var | Looks like |
   |---|---|---|
   | Project URL | `NEXT_PUBLIC_SUPABASE_URL` | `https://abcdefghij.supabase.co` |
   | `anon` `public` | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJhbGci…` JWT, ~200+ chars |
   | `service_role` `secret` | `SUPABASE_SERVICE_ROLE_KEY` | `eyJhbGci…` JWT, ~200+ chars |

   ⚠️ The `service_role` key bypasses RLS. Never embed in client code, never commit.

6. **(Optional but recommended before public launch)** **Authentication → SMTP Settings**: point at Resend/SendGrid. The default email service caps magic-link sends at ~4/hour per user — a real user will hit this.

---

## 3. Stripe — 5 minutes

1. Sign up at <https://dashboard.stripe.com>. Confirm the toggle at the top-left reads **Test mode** for now.
2. **Products → + Add product** (do this twice, both as **one-time** prices in EUR):

   | Name | Price | Env var |
   |---|---|---|
   | SafeCommute Site Unlock — unlimited runs, one site | 100 EUR | `STRIPE_PRICE_SUBSCRIPTION` |
   | SafeCommute Per-run credit | 23 EUR | `STRIPE_PRICE_PER_RUN` |

   Open each product after creating and copy the `price_1...` id (NOT the `prod_1...` id).

   > Both tiers are **one-time payments by design** — there is no recurring charge and no renewal. €100 unlocks unlimited fine-tune runs on one site forever (`subscription_active = true` on `entitlements`). €23 increments `per_run_credits` by one. The Stripe product labelled "Subscription" is a product name only; its price is configured as one-time mode. Repeat revenue comes from new-site unlocks and per-run top-ups — do not flip either product to Stripe recurring mode without an explicit product decision.
   >
   > The env var is still named `STRIPE_PRICE_SUBSCRIPTION` for historical reasons; future refactor should rename it to `STRIPE_PRICE_SITE_UNLOCK`.

3. **Developers → API keys** — **Reveal test key** and use the copy icon (don't click-and-drag, displayed text is sometimes truncated):

   | Stripe key | Vercel env var | Looks like |
   |---|---|---|
   | Secret key | `STRIPE_SECRET_KEY` | `sk_test_...` |
   | Publishable key | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | `pk_test_...` |

4. **Developers → Webhooks → + Add endpoint**:
   - Endpoint URL: `https://safecommute-ai.vercel.app/api/stripe/webhook`
   - Events: search `checkout.session.completed`, select it, Continue.
   - After the endpoint exists, click it and **Reveal** the signing secret (`whsec_...`) → `STRIPE_WEBHOOK_SECRET`.

You should now have 5 Stripe values added to your scratch file next to the 3 Supabase values.

---

## 4. Vercel — 2 minutes

### 4.1 First-time setup

```bash
npx vercel login            # device-flow login, one-time per machine
npx vercel link             # creates / links the project to this directory
```

**When `vercel link` prompts you:**

- Set up and deploy? → `Y`
- Which scope? → your personal account
- Link to existing project? → `Y` if you're re-linking, `N` for first time
- Project name? → `safecommute-ai` (lowercase, no `---`, ≤100 chars). Do NOT accept the default if it contains uppercase or underscores.
- Code directory? → `./` (just Enter — you're already inside `web/`)
- Modify settings? → `N`

> **Critical: never run `vercel` from `~` / home directory.** If the prompt says `You are deploying your home directory`, answer `N` and `cd` into `web/`. A home-directory deploy uploads your entire home folder, which is catastrophic.

### 4.2 Set environment variables

You need **9** vars. For each:

```bash
npx vercel env add <NAME>
```

**When it prompts you:**

- "What's the value?" → paste from your scratch file
- "Which environments?" → space-select **all three** (Production + Preview + Development), Enter. (If you only pick Production, `vercel env pull` without `--environment=production` will return empty values.)
- "Make it sensitive?" → **see the table below**:

   | Env var | Sensitive? | Notes |
   |---|---|---|
   | `NEXT_PUBLIC_SUPABASE_URL` | **No** | Already in client bundle |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | **No** | Already in client bundle; RLS is what secures you |
   | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | **No** | Already in client bundle |
   | `NEXT_PUBLIC_SITE_URL` | **No** | Just a URL |
   | `STRIPE_PRICE_SUBSCRIPTION` | No | Just an id |
   | `STRIPE_PRICE_PER_RUN` | No | Just an id |
   | `SUPABASE_SERVICE_ROLE_KEY` | **Yes** ⚠️ | Bypasses RLS |
   | `STRIPE_SECRET_KEY` | **Yes** | |
   | `STRIPE_WEBHOOK_SECRET` | **Yes** | |

> **Rule of thumb:** `NEXT_PUBLIC_*` → never sensitive (they're already public in the bundle; "sensitive" just prevents you from inspecting them later via `vercel env pull`).

The 9 commands in order (run one at a time — **do NOT pass the value as a positional argument**, the CLI treats the 2nd positional arg as the environment name and errors out):

```bash
npx vercel env add NEXT_PUBLIC_SUPABASE_URL
npx vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY
npx vercel env add SUPABASE_SERVICE_ROLE_KEY
npx vercel env add STRIPE_SECRET_KEY
npx vercel env add STRIPE_WEBHOOK_SECRET
npx vercel env add STRIPE_PRICE_SUBSCRIPTION
npx vercel env add STRIPE_PRICE_PER_RUN
npx vercel env add NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
npx vercel env add NEXT_PUBLIC_SITE_URL     # value: https://safecommute-ai.vercel.app
```

### 4.3 Deploy

```bash
npx vercel --prod
```

Returns the production URL. First `vercel` call creates a preview; `--prod` promotes.

> **`NEXT_PUBLIC_*` env vars are baked into the client JS bundle at build time**, not read at runtime. After changing any `NEXT_PUBLIC_*` value, you **must redeploy** — a simple page reload won't pick up the change.

### 4.4 Verify

```bash
curl -sI https://safecommute-ai.vercel.app/ | head -2
curl -sI https://safecommute-ai.vercel.app/demo/safecommute-v2-demo.zip | grep -iE "^(http|content-length)"
curl -sI https://safecommute-ai.vercel.app/dashboard | grep -iE "^(http|location)"
```

Expect: `200` on landing, `200` + `content-length: 6813037` on the zip, `307 /sign-in?next=/dashboard` on dashboard.

---

## 5. What ships without external services

Even before you wire Supabase + Stripe, the deployment works partially:

- Landing page: **works**.
- Demo download: **works** (static file).
- `/dashboard`, `/sign-in`: **show the "Dashboard offline" Bauhaus error card** — `isSupabaseConfigured()` in `lib/supabase/server.ts` short-circuits with a polished fallback instead of 500-ing.

So you can ship the public URL first, then layer the paid flow on.

---

## 6. Fine-tune worker (still stubbed)

The dashboard queues jobs — it doesn't run them. Jobs stay in `status='queued'` indefinitely until a worker picks them up.

To wire a real worker:

1. Deploy a Python container on Railway / Fly / a cheap GPU VM that, on a loop:
   - polls `finetune_jobs where status='queued'` using the service-role key,
   - updates the row to `status='running'`, sets `started_at = now()`,
   - downloads the matching `audio_clips` rows from `audio-uploads/<owner>/<site_id>/...`,
   - runs `safecommute/pipeline/finetune.py --environment <site.name> --ambient-dir <tempdir> --freeze-cnn`,
   - uploads the produced `{environment}_model.pth` to a new Storage bucket `fine-tuned-models/<owner>/<site_id>/<job_id>.pth`,
   - updates `finetune_jobs` → `status='succeeded'`, `model_path=<storage-path>`, `completed_at=now()`. On failure set `status='failed'` + `error=<exception>`.
2. Add a `/api/finetune/[id]/download` route that validates the job belongs to the user and returns a signed URL from `fine-tuned-models`. The `JobCard` component already renders a Download button when `status === 'succeeded'`.

Until that's in place, paying users see a "Queued" card with the rotating-ring animation — polished enough for demos and early sales calls, not enough for real paying customers.

---

## 7. Returning to an existing deploy

If a Vercel project already exists, check its state before touching anything:

```bash
cd web
npx vercel whoami                       # confirm logged in
npx vercel link                         # relink if .vercel/ is missing
npx vercel env ls                       # list all env vars by name + environment
npx vercel ls                           # list recent deployments
```

To inspect current env var **values** (only for non-sensitive vars):

```bash
npx vercel env pull .env.local --environment=production
awk -F= '/NEXT_PUBLIC_|STRIPE_PRICE/ { print $1, "len="length($2), "prefix="substr($2,1,18) }' .env.local
rm .env.local                           # ALWAYS delete after inspection
```

Sensitive vars (`SUPABASE_SERVICE_ROLE_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`) will pull as empty strings — that's by design, not a bug.

To rotate any var:

```bash
npx vercel env rm <NAME>                # select all envs, confirm
npx vercel env add <NAME>               # paste fresh, all envs, pick sensitivity from the table in §4.2
npx vercel --prod                       # redeploy so NEXT_PUBLIC_* updates reach the client bundle
```

---

## 8. Troubleshooting

### "Invalid API key" on `/sign-in`
Supabase auth error. Cause: `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` don't match — either the anon key was truncated on paste, or one of them points to a different Supabase project.
Fix: copy both fresh from Supabase → Settings → API (use the copy icon, not drag-select). `vercel env rm` both, `vercel env add` both with `sensitive=No`, `vercel --prod`. Verify with `vercel env pull` that the anon key is ~200+ chars and starts `eyJhbGci`.

### Stripe checkout shows "checkout failed" red text
Serverless function 500. Check Vercel function logs:
```bash
npx vercel logs https://safecommute-ai.vercel.app       # streams new events; click "Pay €23" once to trigger
```
Common cause: `StripeAuthenticationError: Invalid API Key provided: sk_test_...` → the `STRIPE_SECRET_KEY` env var is wrong/truncated. Fix:
```bash
npx vercel env rm STRIPE_SECRET_KEY
npx vercel env add STRIPE_SECRET_KEY           # paste fresh, sensitive=Yes
npx vercel --prod
```
Other cause: `No such price: 'price_...'` → `STRIPE_PRICE_SUBSCRIPTION` / `STRIPE_PRICE_PER_RUN` belongs to a different Stripe account. Confirm you're in Test mode in both Stripe and Vercel env vars.

Third cause (**observed 2026-04-21**): `No such price: 'prod_...'` — you pasted a **product id** (`prod_…`) instead of a **price id** (`price_…`). Stripe's `line_items[0].price` only accepts price ids. In Stripe Dashboard → Products → click the product → scroll to the **Pricing** section → copy the id that starts with `price_1…`. Then rotate both `STRIPE_PRICE_*` vars and redeploy:
```bash
npx vercel env rm STRIPE_PRICE_PER_RUN
npx vercel env add STRIPE_PRICE_PER_RUN          # paste price_1... sensitive=No
npx vercel env rm STRIPE_PRICE_SUBSCRIPTION
npx vercel env add STRIPE_PRICE_SUBSCRIPTION     # paste price_1... sensitive=No
npx vercel --prod
```

### "Email rate limit exceeded" at sign-in
Supabase's built-in email service caps at ~4 sends/hour per IP+email. Either wait an hour, stay signed in (sessions persist so no new link needed), or configure a custom SMTP provider (Resend/SendGrid) in Supabase → Auth → SMTP Settings. Do this before public launch.

### Dashboard shows "Dashboard offline" card after deploy
`NEXT_PUBLIC_SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_ANON_KEY` not set in Vercel for the environment being served. Run `npx vercel env ls` to confirm both exist for Production (and ideally all 3 envs), then `npx vercel --prod` to redeploy.

### `.env.local` is all empty strings after `vercel env pull`
Two possible causes:
1. You pulled the wrong environment — default is Development, but your vars are set only for Production. Use `npx vercel env pull .env.local --environment=production`.
2. All your vars are marked sensitive. Sensitive values return as empty strings — this is by design, not a bug. They still work at runtime. Re-add as `sensitive=No` only if you genuinely need to inspect values (and only for non-secret vars like `NEXT_PUBLIC_*`).

### `Error: Environment Variable ... has custom environment ids that do not exist`
You passed the value as a positional arg: `vercel env add NAME some-value`. The CLI reads arg-2 as an environment name. Always run `vercel env add NAME` alone and paste the value at the interactive prompt.

### `vercel logs` exits immediately with nothing
It's a follow stream. Trigger the failing request (e.g. click the checkout button) while the stream is open, or use `npx vercel inspect <deployment-url>` for static info about the build.

### Sign-in succeeds but Stripe webhook doesn't grant credits
The webhook at `/api/stripe/webhook` wasn't wired, or the `STRIPE_WEBHOOK_SECRET` doesn't match. Go to Stripe → Webhooks → click your endpoint → **Send test webhook** → event `checkout.session.completed`. Then check Vercel logs. Signing-secret mismatch shows as `invalid signature` in the response.

### Webhook delivery shows `200 OK` in Stripe but `payments` and `entitlements` stay empty in Supabase (**observed 2026-04-21**)
The route returned 200 without throwing — but the two `upsert` calls in [app/api/stripe/webhook/route.ts](web/app/api/stripe/webhook/route.ts) are **not error-checked**, so a silently-blocked insert still returns 200 to Stripe. Root cause is almost always that `SUPABASE_SERVICE_ROLE_KEY` is set to the **`anon`** key instead of the **`service_role`** key (both are `eyJhbGci…` JWTs ~200 chars, trivial to paste-slip). With the anon key, RLS blocks inserts into `payments` / `entitlements` because the schema allows only self-read, no self-insert — only `service_role` bypasses RLS.

**Fix:**

1. In Supabase Dashboard → **Project Settings → API**, copy the value under **`service_role` `secret`** (not `anon`). Confirm by pasting the JWT's middle segment into <https://jwt.io> — payload must read `"role":"service_role"`.
2. Rotate the Vercel env var:
   ```bash
   cd web
   npx vercel env rm SUPABASE_SERVICE_ROLE_KEY     # all envs, y
   npx vercel env add SUPABASE_SERVICE_ROLE_KEY    # paste fresh, sensitive=Yes, all 3 envs
   npx vercel --prod
   ```
3. Wait for the deployment to show `Ready`, then **Stripe → Webhooks → your endpoint → Event deliveries →** click the stuck `checkout.session.completed` row → **Resend** (top right). The replay hits the fixed deployment and grants the credit without re-charging the card.

After the resend succeeds, Supabase `payments` has a new row with `status='paid'` and `entitlements.per_run_credits` bumps by 1 (or `subscription_active=true` for the €100 tier).

---

## 9. Post-deploy smoke test

Status as of **2026-04-21** on `https://safecommute-ai.vercel.app`: paid flow is end-to-end green. Resume from the first unchecked item.

- [x] Landing in incognito — animations render, Bauhaus disc follows cursor.
- [x] "Download demo ↓" → 6.5 MB zip → unzipped contains `safecommute_v2.pth` + `short.md`.
- [x] "Open dashboard →" → redirects to `/sign-in` (not the offline card).
- [x] Enter email → magic link lands in inbox → clicking returns to `/dashboard`.
- [x] `/dashboard/billing` → "Pay €23 →" → Stripe Checkout → card `4242 4242 4242 4242` any future date any CVC → returns with "Payment received" banner + `1 credit`. *(Required two fixes: §3.2 `prod_` → `price_` for both Stripe price env vars, and §8 `SUPABASE_SERVICE_ROLE_KEY` rotation so the webhook writes weren't dropped.)*
- [x] Supabase Table Editor shows rows in `payments`, `entitlements`.
- [ ] New site → drop 3 WAVs → "Run fine-tune" → queued job card appears, credit drops to 0.
- [ ] Supabase shows rows in `sites`, `audio_clips`, `finetune_jobs`; files in Storage → `audio-uploads/<uid>/<site-id>/`.

After the two remaining items pass, the job will stay at `status='queued'` forever — that is expected. §6 (fine-tune worker + `/api/finetune/[id]/download` route + `fine-tuned-models` Storage bucket) is the next workstream and the last blocker before a paying customer can complete a flow.

### Env var hygiene (found 2026-04-21)

Partial drift across environments — doesn't block production but will break preview deploys:

- Prod **and** Preview: `STRIPE_SECRET_KEY`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_SITE_URL`.
- Prod only (missing from Preview): `STRIPE_WEBHOOK_SECRET`, `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`, `STRIPE_PRICE_PER_RUN`, `STRIPE_PRICE_SUBSCRIPTION`, `SUPABASE_SERVICE_ROLE_KEY`.
- **None set for Development.** `npx vercel env pull .env.local` without `--environment=production` returns all empty strings.

When you need preview URLs working (e.g. PR preview deploys), re-add each Prod-only var with all three environment targets selected.
