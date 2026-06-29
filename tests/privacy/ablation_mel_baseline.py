"""
Mel-vs-PCEN ablation runner. Runs `run_attack_eval.py` for the two
`mel_baseline` configurations only, so the comparison "how much privacy
does PCEN add over plain mel" can be re-run cheaply without re-rendering
the PCEN paths.

Output: tests/privacy/reports/ablation_mel_baseline.json with the same
schema as `raw_results.json` but only the mel-baseline rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tests/privacy/data/librispeech_devclean_3s")
    parser.add_argument("--out", default="tests/privacy/reports/ablation_mel_baseline.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    tmp_dir = out_path.parent / "_ablation_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "tests/privacy/run_attack_eval.py",
        "--corpus", args.corpus,
        "--out-dir", str(tmp_dir),
        "--configurations",
        "mel_baseline:griffin_lim",
        "mel_baseline:hifigan",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    subprocess.check_call(cmd, env={**__import__("os").environ, "PYTHONPATH": "."})

    raw = json.loads((tmp_dir / "raw_results.json").read_text())
    out_path.write_text(json.dumps(raw, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
