"""SafeCommute fine-tune worker loop.

Polls the Supabase `finetune_jobs` table for queued jobs, runs each one
through worker.job.process_job, and loops. Designed for WORKER_CONCURRENCY=1
on the Ryzen 7 7435HS dev box (see the Phase 2 finalization plan) — parallel
runs would thrash the CPU since each fine-tune takes ~10–20 minutes.

Deploy as a systemd user unit (see systemd/safecommute-worker.service).

Env:
  SUPABASE_URL                   (or NEXT_PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_ROLE_KEY
  WORKER_POLL_INTERVAL_S         default 15
  CLIPS_BUCKET / MODEL_BUCKET    override bucket names if they drift
  FT_KEEP_SAFE_RATIO / FT_EPOCHS / FT_LR / MAJORITY_K   recipe overrides

Run locally:
    python -m worker.main
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time

from .env import env_int
from .job import run_job_safely
from .supabase_client import Supabase

POLL_INTERVAL_S = env_int("WORKER_POLL_INTERVAL_S", 15)
SHUTDOWN = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global SHUTDOWN
        logging.getLogger(__name__).info(
            "received %s, finishing current job then exiting",
            signal.Signals(signum).name)
        SHUTDOWN = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("worker")
    log.info("safecommute worker starting (poll=%ss)", POLL_INTERVAL_S)
    _install_signal_handlers()

    sb = Supabase.from_env()

    while not SHUTDOWN:
        try:
            job = sb.claim_next_queued_job()
        except Exception as e:  # noqa: BLE001
            log.exception("poll failed: %s — sleeping", e)
            time.sleep(POLL_INTERVAL_S)
            continue

        if not job:
            time.sleep(POLL_INTERVAL_S)
            continue

        log.info("claimed job %s", job["id"])
        run_job_safely(sb, job)

    log.info("clean shutdown")


if __name__ == "__main__":
    sys.exit(main() or 0)
