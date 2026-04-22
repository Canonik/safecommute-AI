"""Defensive env-var parsing for the worker.

systemd's `EnvironmentFile=` directive does NOT strip inline `# ...`
comments from `KEY=VALUE` lines. A user who carries over dotenv habits
and writes `WORKER_POLL_INTERVAL_S=15   # sleep between polls` will end
up with the env var literally set to `"15   # sleep between polls"`,
and a bare `int(os.environ.get(...))` crashes the service on startup.

These helpers tolerate that by:
  1. stripping any `#` suffix and the trailing whitespace before it,
  2. trimming quotes (another common dotenv-ism systemd preserves literally),
  3. falling back to the default if the result is empty.

Used everywhere the worker parses int/float/str env vars so one bad line
in .env gets reported as a clear error (or silently normalised away)
rather than killing the process loop.
"""
from __future__ import annotations

import os
from typing import Optional


def _clean(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    value = raw
    # Drop an unquoted inline comment: "15   # sleep ..." -> "15".
    # We intentionally don't try to parse quoted `#` inside values — env
    # values with a real `#` in them should be quoted or URL-encoded.
    if "#" in value:
        # Only strip if there's whitespace before the `#`, to avoid
        # clobbering legitimate `#` characters early in the string.
        head, _, _ = value.partition("#")
        if head != value and head.rstrip() != head:
            value = head
    value = value.strip()
    # Strip one pair of surrounding quotes if present.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    cleaned = _clean(raw)
    return cleaned or default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    cleaned = _clean(raw)
    if not cleaned:
        return default
    try:
        return int(cleaned)
    except ValueError as e:
        raise ValueError(
            f"env var {name}={raw!r} is not an int "
            f"(parsed as {cleaned!r}). If you copied worker/.env.example, "
            f"remove any `# comment` after the value — systemd does not "
            f"strip inline comments."
        ) from e


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    cleaned = _clean(raw)
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError as e:
        raise ValueError(
            f"env var {name}={raw!r} is not a float "
            f"(parsed as {cleaned!r}). See worker/.env.example for formatting."
        ) from e
