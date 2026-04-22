-- 0002_models_deliverable_bucket.sql
-- Companion to 0001_init.sql. Adds the output bucket the background worker
-- (worker/main.py) writes fine-tune artefacts to, with RLS that scopes reads
-- to {owner}/<...> object paths.
--
-- Run after 0001_init.sql (e.g. `supabase db push` or paste into the SQL editor).
-- Idempotent — `on conflict do update` on the bucket row, `drop if exists` on
-- policies.

-- ---------------------------------------------------------------------------
-- MODELS-DELIVERABLE — private bucket for fine-tuned artefacts.
-- Worker writes three files per succeeded job under
--   {owner_id}/{job_id}/model.onnx
--   {owner_id}/{job_id}/thresholds.json
--   {owner_id}/{job_id}/deployment_report.json
-- No size_limit per-file (artefacts are ~4 MB INT8 ONNX + small JSONs), but
-- we scope allowed_mime_types for defence-in-depth.
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('models-deliverable', 'models-deliverable', false, 52428800,
        array['application/octet-stream', 'application/json',
              'application/x-onnx', 'model/onnx'])
on conflict (id) do update set
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- Users can list / read their own artefacts; worker writes as service role
-- and bypasses RLS, so we only need a self-read policy for downloads.
drop policy if exists "deliverable self-read" on storage.objects;
create policy "deliverable self-read" on storage.objects
  for select using (
    bucket_id = 'models-deliverable'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

-- No user-write policy on purpose — only the worker (service role) writes.
-- No user-delete policy on purpose — paid artefact lifecycle is manual.

-- ---------------------------------------------------------------------------
-- WORKER_LOGS — structured error traces the worker writes when a job fails.
-- Kept separate from finetune_jobs.error so one-line customer-facing messages
-- can live in the job row and noisy tracebacks stay out of the dashboard.
-- ---------------------------------------------------------------------------
create table if not exists public.worker_logs (
  id         bigserial primary key,
  job_id     uuid not null references public.finetune_jobs(id) on delete cascade,
  level      text not null check (level in ('info','warn','error')),
  message    text not null,
  traceback  text,
  created_at timestamptz not null default now()
);
create index if not exists worker_logs_job_idx on public.worker_logs(job_id, created_at desc);

alter table public.worker_logs enable row level security;

-- Owners of the parent job can read their own logs (useful for support / retry).
drop policy if exists "worker_logs self-read" on public.worker_logs;
create policy "worker_logs self-read" on public.worker_logs
  for select using (
    exists (
      select 1 from public.finetune_jobs j
      where j.id = worker_logs.job_id and j.owner = auth.uid()
    )
  );

-- No user insert/update/delete — service role only.
