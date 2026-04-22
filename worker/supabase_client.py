"""Thin Supabase REST + Storage helper for the fine-tune worker.

Why not supabase-py? The worker only needs:
  - UPDATE finetune_jobs ... WHERE id=? AND status='queued' (CAS, via PostgREST)
  - list / download objects from audio-uploads
  - upload objects to models-deliverable
  - delete objects from audio-uploads (privacy-fix §2.3)
  - insert into payments-adjacent tables (worker_logs)

All of that is a handful of HTTPS calls to the Supabase REST endpoints. Pulling
in supabase-py and its postgrest-py transitive deps is overkill for 200 lines
of worker logic. We use `requests` directly, which is already a dep of
yt-dlp/torchaudio anyway.

Authentication uses the service-role key (from SUPABASE_SERVICE_ROLE_KEY) which
bypasses RLS — *do not* expose this helper to any code that runs inside a
user-authenticated request.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


@dataclass
class Supabase:
    url: str                     # e.g. https://<project>.supabase.co
    service_role_key: str

    @classmethod
    def from_env(cls) -> "Supabase":
        url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        return cls(url=url.rstrip("/"), service_role_key=key)

    # ------------------------------------------------------------------ REST
    @property
    def _auth_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
        }

    def _rest(self, path: str, *, method: str = "GET",
              params: Optional[Dict[str, Any]] = None,
              json: Optional[Any] = None,
              prefer: Optional[str] = None) -> requests.Response:
        headers = dict(self._auth_headers)
        headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        url = f"{self.url}/rest/v1{path}"
        r = requests.request(method, url, headers=headers,
                             params=params, json=json, timeout=30)
        if r.status_code >= 400:
            log.error("supabase rest %s %s -> %s: %s",
                      method, path, r.status_code, r.text[:400])
        return r

    def claim_next_queued_job(self) -> Optional[Dict[str, Any]]:
        """Atomic CAS: set status='running' on the oldest queued job.

        Postgres-level atomicity via a single UPDATE...RETURNING filtered by
        status='queued'. If two workers call this concurrently, only one
        gets a row back.
        """
        params = {
            "status": "eq.queued",
            "order": "created_at.asc",
            "limit": 1,
            "select": "id,site_id,owner,created_at",
        }
        r = self._rest("/finetune_jobs",
                       method="PATCH",
                       params=params,
                       json={"status": "running",
                             "started_at": "now()"},
                       prefer="return=representation")
        if r.status_code >= 400:
            return None
        rows = r.json() or []
        return rows[0] if rows else None

    def mark_job_succeeded(self, job_id: str, *,
                           model_path: str,
                           thresholds: Dict[str, Any]) -> None:
        r = self._rest("/finetune_jobs",
                       method="PATCH",
                       params={"id": f"eq.{job_id}"},
                       json={"status": "succeeded",
                             "completed_at": "now()",
                             "model_path": model_path,
                             "thresholds": thresholds,
                             "error": None},
                       prefer="return=minimal")
        r.raise_for_status()

    def mark_job_failed(self, job_id: str, *,
                        error: str) -> None:
        r = self._rest("/finetune_jobs",
                       method="PATCH",
                       params={"id": f"eq.{job_id}"},
                       json={"status": "failed",
                             "completed_at": "now()",
                             "error": error[:500]},
                       prefer="return=minimal")
        r.raise_for_status()

    def list_clips_for_site(self, site_id: str) -> List[Dict[str, Any]]:
        r = self._rest("/audio_clips",
                       params={"site_id": f"eq.{site_id}",
                               "select": "id,storage_path,filename,size_bytes",
                               "order": "uploaded_at.asc"})
        r.raise_for_status()
        return r.json()

    def get_site(self, site_id: str) -> Dict[str, Any]:
        r = self._rest("/sites",
                       params={"id": f"eq.{site_id}",
                               "select": "id,owner,name,environment",
                               "limit": 1})
        r.raise_for_status()
        rows = r.json()
        if not rows:
            raise RuntimeError(f"site {site_id} not found")
        return rows[0]

    def insert_worker_log(self, job_id: str, *, level: str,
                          message: str, traceback: Optional[str] = None) -> None:
        r = self._rest("/worker_logs",
                       method="POST",
                       json={"job_id": job_id, "level": level,
                             "message": message[:1000],
                             "traceback": traceback},
                       prefer="return=minimal")
        if r.status_code >= 400:
            log.warning("failed to insert worker_log: %s", r.text[:200])

    # --------------------------------------------------------------- Storage
    def storage_download(self, bucket: str, path: str, dest: str) -> None:
        """Download an object from a private bucket via the storage API."""
        url = f"{self.url}/storage/v1/object/{bucket}/{path}"
        r = requests.get(url, headers=self._auth_headers, stream=True,
                         timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(
                f"storage download failed for {bucket}/{path}: "
                f"{r.status_code} {r.text[:200]}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)

    def storage_upload(self, bucket: str, path: str, local_path: str,
                       content_type: str) -> None:
        """Upload (or overwrite — `x-upsert: true`) an object."""
        url = f"{self.url}/storage/v1/object/{bucket}/{path}"
        headers = dict(self._auth_headers)
        headers["Content-Type"] = content_type
        headers["x-upsert"] = "true"
        with open(local_path, "rb") as f:
            r = requests.post(url, headers=headers, data=f, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(
                f"storage upload failed for {bucket}/{path}: "
                f"{r.status_code} {r.text[:200]}")

    def storage_delete(self, bucket: str, paths: List[str]) -> None:
        """Delete multiple objects. Privacy-fix §2.3: called on job success
        to wipe the raw-audio uploads from audio-uploads after the INT8 model
        has been produced."""
        if not paths:
            return
        url = f"{self.url}/storage/v1/object/{bucket}"
        headers = dict(self._auth_headers)
        headers["Content-Type"] = "application/json"
        r = requests.delete(url, headers=headers,
                            json={"prefixes": paths}, timeout=30)
        if r.status_code >= 400:
            log.warning("storage delete failed for %s: %s %s",
                        bucket, r.status_code, r.text[:200])
