-- SafeCommute dashboard schema.
-- Run once in Supabase SQL editor (or via `supabase db push` if you use the CLI).
-- Idempotent-ish: uses IF NOT EXISTS / OR REPLACE where supported.

-- ---------------------------------------------------------------------------
-- PROFILES — mirrors auth.users, lets us attach app-level fields safely.
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
  id         uuid primary key references auth.users(id) on delete cascade,
  email      text,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "profile self-read" on public.profiles;
create policy "profile self-read" on public.profiles
  for select using (auth.uid() = id);

drop policy if exists "profile self-upsert" on public.profiles;
create policy "profile self-upsert" on public.profiles
  for insert with check (auth.uid() = id);

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.profiles (id, email) values (new.id, new.email)
  on conflict (id) do nothing;
  insert into public.entitlements (owner) values (new.id)
  on conflict (owner) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- ENTITLEMENTS — what the user has paid for.
-- ---------------------------------------------------------------------------
create table if not exists public.entitlements (
  owner                uuid primary key references auth.users(id) on delete cascade,
  subscription_active  boolean not null default false,
  per_run_credits      int not null default 0,
  updated_at           timestamptz not null default now()
);

alter table public.entitlements enable row level security;

drop policy if exists "entitlement self-read" on public.entitlements;
create policy "entitlement self-read" on public.entitlements
  for select using (auth.uid() = owner);

-- ---------------------------------------------------------------------------
-- SITES — a fine-tune target (e.g. "Milano Centrale platform 4").
-- ---------------------------------------------------------------------------
create table if not exists public.sites (
  id          uuid primary key default gen_random_uuid(),
  owner       uuid not null references auth.users(id) on delete cascade,
  name        text not null,
  environment text not null check (environment in ('metro','retail','school','eldercare','industrial','other')),
  created_at  timestamptz not null default now()
);
create index if not exists sites_owner_idx on public.sites(owner);

alter table public.sites enable row level security;

drop policy if exists "sites self-crud" on public.sites;
create policy "sites self-crud" on public.sites
  for all using (auth.uid() = owner) with check (auth.uid() = owner);

-- ---------------------------------------------------------------------------
-- AUDIO_CLIPS — per-site uploads stored in Storage bucket `audio-uploads`.
-- ---------------------------------------------------------------------------
create table if not exists public.audio_clips (
  id           uuid primary key default gen_random_uuid(),
  site_id      uuid not null references public.sites(id) on delete cascade,
  owner        uuid not null references auth.users(id) on delete cascade,
  storage_path text not null,
  filename     text not null,
  size_bytes   bigint,
  duration_s   double precision,
  uploaded_at  timestamptz not null default now()
);
create index if not exists clips_site_idx on public.audio_clips(site_id);
create index if not exists clips_owner_idx on public.audio_clips(owner);

alter table public.audio_clips enable row level security;

drop policy if exists "clips self-crud" on public.audio_clips;
create policy "clips self-crud" on public.audio_clips
  for all using (auth.uid() = owner) with check (auth.uid() = owner);

-- ---------------------------------------------------------------------------
-- FINETUNE_JOBS — queued/running/succeeded/failed.
-- ---------------------------------------------------------------------------
create table if not exists public.finetune_jobs (
  id           uuid primary key default gen_random_uuid(),
  site_id      uuid not null references public.sites(id) on delete cascade,
  owner        uuid not null references auth.users(id) on delete cascade,
  status       text not null check (status in ('queued','running','succeeded','failed')),
  model_path   text,
  thresholds   jsonb,
  error        text,
  created_at   timestamptz not null default now(),
  started_at   timestamptz,
  completed_at timestamptz
);
create index if not exists jobs_owner_idx on public.finetune_jobs(owner, created_at desc);
create index if not exists jobs_site_idx on public.finetune_jobs(site_id, created_at desc);

alter table public.finetune_jobs enable row level security;

drop policy if exists "jobs self-read" on public.finetune_jobs;
create policy "jobs self-read" on public.finetune_jobs
  for select using (auth.uid() = owner);

drop policy if exists "jobs self-insert" on public.finetune_jobs;
create policy "jobs self-insert" on public.finetune_jobs
  for insert with check (auth.uid() = owner);

-- updates to jobs come from service-role worker only — no user update policy.

-- ---------------------------------------------------------------------------
-- PAYMENTS — Stripe session ledger.
-- ---------------------------------------------------------------------------
create table if not exists public.payments (
  id                 uuid primary key default gen_random_uuid(),
  owner              uuid not null references auth.users(id) on delete cascade,
  stripe_session_id  text unique not null,
  kind               text not null check (kind in ('subscription','per_run')),
  amount_cents       int not null,
  currency           text not null default 'eur',
  status             text not null default 'pending',
  created_at         timestamptz not null default now()
);
create index if not exists payments_owner_idx on public.payments(owner, created_at desc);

alter table public.payments enable row level security;

drop policy if exists "payments self-read" on public.payments;
create policy "payments self-read" on public.payments
  for select using (auth.uid() = owner);

-- Inserts/updates on payments come from the Stripe webhook (service-role).

-- ---------------------------------------------------------------------------
-- STORAGE BUCKET — audio-uploads (private, per-user folder).
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('audio-uploads', 'audio-uploads', false, 26214400, array['audio/wav','audio/x-wav','audio/wave','audio/mpeg','audio/mp4','audio/flac'])
on conflict (id) do update set
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "audio self-read" on storage.objects;
create policy "audio self-read" on storage.objects
  for select using (
    bucket_id = 'audio-uploads'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

drop policy if exists "audio self-write" on storage.objects;
create policy "audio self-write" on storage.objects
  for insert with check (
    bucket_id = 'audio-uploads'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

drop policy if exists "audio self-delete" on storage.objects;
create policy "audio self-delete" on storage.objects
  for delete using (
    bucket_id = 'audio-uploads'
    and (storage.foldername(name))[1] = auth.uid()::text
  );
