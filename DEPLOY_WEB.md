# Deploy the SafeCommute web + dashboard

Ships the marketing site, the demo download, and the paid fine-tune dashboard in one Vercel deployment backed by Supabase (DB + Auth + Storage) and Stripe (payments).

**Current production URL:** <https://safecommute-ai.vercel.app>

All commands assume **`cwd = web/`** (i.e. `cd web` from the repo root first).

> **If you're returning to this project and things already exist** (Supabase project, Vercel project linked, env vars set), skip to **[Section 7 — Returning to an existing deploy](#7-returning-to-an-existing-deploy)** first to orient yourself.

---

## 0. What you'll end up with

- `https://safecommute-ai.vercel.app` — public landing page (Bauhaus).
- `https://safecommute-ai.vercel.app/demo/safecommute-v2-demo.zip` — 6.5 MB demo bundle (model + `short.md`).
- `https://safecommute-ai.vercel.app/dashboard` — authenticated dashboard. Magic-link sign-in, €100 subscription OR €23 per-run via Stripe Checkout, audio clip uploads to Supabase Storage, queued fine-tune jobs.

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
   | SafeCommute Subscription — one site | 100 EUR | `STRIPE_PRICE_SUBSCRIPTION` |
   | SafeCommute Per-run credit | 23 EUR | `STRIPE_PRICE_PER_RUN` |

   Open each product after creating and copy the `price_1...` id (NOT the `prod_1...` id).

   > The "subscription" label is a product name, not a Stripe recurring subscription. Keeps the UX simple: user pays once, dashboard flags `subscription_active = true`, unlimited runs forever for that user. To switch to real recurring later, change the Stripe price to recurring and flip the Checkout `mode` to `"subscription"` in `web/app/api/checkout/route.ts`.

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

---

## 9. Post-deploy smoke test

- [ ] Landing in incognito — animations render, Bauhaus disc follows cursor.
- [ ] "Download demo ↓" → 6.5 MB zip → unzipped contains `safecommute_v2.pth` + `short.md`.
- [ ] "Open dashboard →" → redirects to `/sign-in` (not the offline card).
- [ ] Enter email → magic link lands in inbox → clicking returns to `/dashboard`.
- [ ] `/dashboard/billing` → "Pay €23 →" → Stripe Checkout → card `4242 4242 4242 4242` any future date any CVC → returns with "Payment received" banner + `1 credit`.
- [ ] Supabase Table Editor shows rows in `payments`, `entitlements`.
- [ ] New site → drop 3 WAVs → "Run fine-tune" → queued job card appears, credit drops to 0.
- [ ] Supabase shows rows in `sites`, `audio_clips`, `finetune_jobs`; files in Storage → `audio-uploads/<uid>/<site-id>/`.
