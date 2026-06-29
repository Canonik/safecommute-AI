#!/usr/bin/env python
"""Release checklist for the PCEN reconstruction audit snapshot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    ".gitignore",
    "paper/main.tex",
    "paper/references.bib",
    "paper/preprint.sty",
    "tests/privacy/REPORT.md",
    "tests/privacy/reports/raw_results.json",
    "tests/privacy/reports/hidden_phrases/raw_results.json",
]

EXCLUDED_PATHS = [
    "raw_data",
    "prepared_data",
    "models",
    ".env",
    ".env.local",
    "wandb",
    "runs",
    "logs",
]

EXCLUDED_GLOBS = ["*.pth", "*.onnx", "*.pt"]
SECRET_MARKERS = [
    "api" + "_" + "key",
    "api" + "key",
    "secret" + "_" + "key",
    "stripe" + "_",
    "supa" + "base",
]


def _run(cmd: list[str], cwd: Path = ROOT) -> bool:
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true", help="only run structural release checks")
    parser.add_argument("--latex", action="store_true", help="also compile paper/main.tex")
    args = parser.parse_args()

    failures: list[str] = []
    smoke_ok = True if args.skip_smoke else _run([sys.executable, "scripts/smoke_test.py"])

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            failures.append(f"missing required file: {rel}")

    for rel in EXCLUDED_PATHS:
        if (ROOT / rel).exists():
            failures.append(f"excluded path present: {rel}")

    for pattern in EXCLUDED_GLOBS:
        matches = [
            p for p in ROOT.rglob(pattern)
            if ".git" not in p.parts and "__pycache__" not in p.parts
        ]
        if matches:
            failures.append(
                f"excluded artifact(s) present for {pattern}: "
                + ", ".join(str(p.relative_to(ROOT)) for p in matches[:5])
            )

    text_files = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and p.suffix.lower() in {".py", ".md", ".tex", ".bib", ".txt", ".cff"}
    ]
    for path in text_files:
        if path.resolve() == Path(__file__).resolve():
            continue
        haystack = path.read_text(errors="ignore").lower()
        for marker in SECRET_MARKERS:
            if marker in haystack:
                failures.append(f"possible secret/product marker `{marker}` in {path.relative_to(ROOT)}")

    if not smoke_ok:
        failures.append("smoke test failed")

    if args.latex:
        latex_cmds = [
            ["pdflatex", "main.tex"],
            ["bibtex", "main"],
            ["pdflatex", "main.tex"],
            ["pdflatex", "main.tex"],
        ]
        for cmd in latex_cmds:
            if not _run(cmd, cwd=ROOT / "paper"):
                failures.append("paper compile failed: " + " ".join(cmd))
                break

    checklist = [
        ("no raw_data", not (ROOT / "raw_data").exists()),
        ("no prepared_data", not (ROOT / "prepared_data").exists()),
        ("no model checkpoints", not list(ROOT.rglob("*.pth")) and not list(ROOT.rglob("*.pt"))),
        ("no ONNX exports", not list(ROOT.rglob("*.onnx"))),
        ("no .env", not any((ROOT / name).exists() for name in [".env", ".env.local"])),
        ("paper source present", (ROOT / "paper/main.tex").is_file()),
        ("smoke test passes", smoke_ok),
        ("raw_results.json present", (ROOT / "tests/privacy/reports/raw_results.json").is_file()),
        ("REPORT.md generated from raw_results.json", (ROOT / "tests/privacy/REPORT.md").is_file()),
    ]

    for label, ok in checklist:
        print(f"[{'x' if ok else ' '}] {label}")

    if failures:
        print("\nRelease checklist failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nRelease checklist passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
