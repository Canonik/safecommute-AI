"""
Render tests/privacy/REPORT.md from raw_results.json.

The report has a fixed structure so the paper can cite it stably:
  1. Threat model and method
  2. Per-config metrics table with CIs
  3. mel-vs-PCEN ablation (the load-bearing comparison)
  4. Sample reconstructions
  5. Verdict line, written in pure prose using the measured numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_metric(d: dict, fmt: str = ".3f") -> str:
    return f"{d['mean']:{fmt}}  [{d['ci95_lo']:{fmt}}, {d['ci95_hi']:{fmt}}]  (n={d['n']})"


def _find(configs, recovery, attack):
    for c in configs:
        if c["recovery"] == recovery and c["attack"] == attack:
            return c
    return None


def _verdict(results: dict) -> str:
    configs = results["configurations"]
    pcen_oracle_gl = _find(configs, "pcen_oracle", "griffin_lim")
    pcen_blind_gl = _find(configs, "pcen_blind", "griffin_lim")
    mel_gl = _find(configs, "mel_baseline", "griffin_lim")
    pcen_oracle_hf = _find(configs, "pcen_oracle", "hifigan")
    mel_hf = _find(configs, "mel_baseline", "hifigan")

    lines = []
    lines.append("### Bottom line\n")
    if pcen_oracle_gl is None or mel_gl is None:
        lines.append("Insufficient configurations to draw a verdict.\n")
        return "\n".join(lines)

    pcen_wer = pcen_oracle_gl["metrics"]["wer"]
    mel_wer = mel_gl["metrics"]["wer"]
    delta_gl = pcen_wer["mean"] - mel_wer["mean"]
    if pcen_oracle_hf and mel_hf:
        pcen_wer_hf = pcen_oracle_hf["metrics"]["wer"]
        mel_wer_hf = mel_hf["metrics"]["wer"]
        delta_hf = pcen_wer_hf["mean"] - mel_wer_hf["mean"]
    else:
        delta_hf = None

    lines.append(f"Under the **Griffin-Lim** attack, WER from plain mel is "
                 f"{mel_wer['mean']:.2f} (95 % CI [{mel_wer['ci95_lo']:.2f}, "
                 f"{mel_wer['ci95_hi']:.2f}]); WER from PCEN tiles (oracle attacker) "
                 f"is {pcen_wer['mean']:.2f} (95 % CI [{pcen_wer['ci95_lo']:.2f}, "
                 f"{pcen_wer['ci95_hi']:.2f}]). PCEN minus mel: "
                 f"{delta_gl:+.3f} WER points.\n")
    if delta_hf is not None:
        lines.append(f"Under the **HiFi-GAN** attack, WER from plain mel is "
                     f"{mel_wer_hf['mean']:.2f}; WER from PCEN (oracle) is "
                     f"{pcen_wer_hf['mean']:.2f}. PCEN minus mel: "
                     f"{delta_hf:+.3f} WER points.\n")

    if pcen_blind_gl:
        bw = pcen_blind_gl["metrics"]["wer"]
        lines.append(f"The **blind** PCEN attacker (no oracle running-mean) "
                     f"achieves WER {bw['mean']:.2f} (95 % CI "
                     f"[{bw['ci95_lo']:.2f}, {bw['ci95_hi']:.2f}]) under "
                     f"Griffin-Lim.\n")

    # Plain-text verdict
    high_wer_threshold = 0.85
    cos_threshold_above_chance_pp = 0.10
    pcen_oracle_block_speech = pcen_wer["ci95_lo"] >= high_wer_threshold
    cos = pcen_oracle_gl["metrics"]["speaker_cosine"]
    cos_chance = pcen_oracle_gl["metrics"]["speaker_cosine_chance"]
    cos_above_chance = (cos["mean"] - cos_chance["mean"]) > cos_threshold_above_chance_pp

    lines.append("\n**Interpretation:**\n")
    if pcen_oracle_block_speech and not cos_above_chance:
        lines.append("- PCEN tiles do not yield intelligible speech under either "
                     "attack in this run; speaker cosine stays at chance.\n")
    elif not pcen_oracle_block_speech:
        lines.append("- PCEN tiles yield partially intelligible speech under "
                     f"the strongest attacker (WER lower bound "
                     f"{pcen_wer['ci95_lo']:.2f}). PCEN should not be treated as "
                     "privacy-preserving or non-invertible.\n")
    if cos_above_chance:
        lines.append(f"- Speaker identity leaks: reconstructed cosine "
                     f"({cos['mean']:.2f}) exceeds chance "
                     f"({cos_chance['mean']:.2f}) by "
                     f"{cos['mean'] - cos_chance['mean']:+.2f}.\n")
    return "\n".join(lines)


def _hidden_phrase_section(hidden_path: Path) -> list[str]:
    """Render the hidden-phrase sub-evaluation, if results exist."""
    if not hidden_path.exists():
        return []
    data = json.loads(hidden_path.read_text())
    lines: list[str] = ["", "## Hidden-phrase sub-evaluation", ""]
    lines.append(
        f"A separate corpus of {data['n_clips']} synthesised probe clips "
        f"(speechbrain Tacotron2 + HiFi-GAN on LJSpeech, resampled to "
        f"{data['sample_rate']} Hz) was generated with planted phrases "
        f"such as `\"the password is fortepiano\"`. We push each clip "
        f"through the same six (recovery x attack) pipeline and report "
        f"per-clip Whisper transcripts so a reader can spot-check whether "
        f"specific keywords survive."
    )
    lines.append("")
    lines.append("| Configuration | Planted phrase | Whisper hypothesis |")
    lines.append("|---|---|---|")
    for c in data["configurations"]:
        for ex in c["examples"]:
            ref = ex["reference"][:60].replace("|", "/")
            hyp = ex["hypothesis"][:60].replace("|", "/")
            lines.append(
                f"| `{c['recovery']}/{c['attack']}` | {ref} | {hyp} |"
            )
    lines.append("")
    lines.append("Aggregate metrics for the probe-phrase corpus (n is small, so CIs are wide):")
    lines.append("")
    lines.append("| Recovery | Attack | WER | Speaker cosine | PESQ-wb | STOI |")
    lines.append("|---|---|---|---|---|---|")
    for c in data["configurations"]:
        m = c["metrics"]
        lines.append(
            f"| `{c['recovery']}` | `{c['attack']}` "
            f"| {_fmt_metric(m['wer'])} "
            f"| {_fmt_metric(m['speaker_cosine'])} "
            f"| {_fmt_metric(m['pesq'])} "
            f"| {_fmt_metric(m['stoi'])} |"
        )
    return lines


def render(raw_path: Path, out_path: Path, hidden_path: Path | None = None):
    results = json.loads(raw_path.read_text())
    configs = results["configurations"]

    lines: list[str] = []
    a = lines.append

    a(f"# Privacy attack evaluation -- SafeCommute PCEN tiles\n")
    a(f"Corpus: `{results['corpus']}`, n={results['n_clips']} clips, "
      f"{results['sample_rate']} Hz.\n")
    a("Threat model: the attacker observes the PCEN tile the classifier "
      "consumes plus the public feature-extraction config "
      "(`librosa.pcen` defaults, prescale `2**31`). They try to recover "
      "intelligible audio or speaker identity.\n")
    a("**Recoveries**\n")
    a("- `pcen_oracle`: exact closed-form inverse given the running-mean state "
      "M from the forward pass. Strongest attacker upper bound.\n")
    a("- `pcen_blind`: iterative inverse, M estimated from the PCEN output itself. "
      "Realistic attacker.\n")
    a("- `mel_baseline`: feed the raw mel directly to the attack, skipping PCEN. "
      "Ablation: quantifies how much privacy PCEN adds over plain mel.\n")
    a("**Attacks**\n")
    a("- `griffin_lim`: power-mel -> waveform via 60 iterations of "
      "Griffin-Lim phase estimation. No pretrained model.\n")
    a("- `hifigan`: 64-mel -> 80-mel (closed-form pinv adapter) -> log-mel "
      "-> SpeechBrain `tts-hifigan-libritts-16kHz` vocoder. "
      "Off-the-shelf, no fine-tuning.\n")

    a("## Metrics per configuration\n")
    a("All values are mean with bootstrap 95 % CI in brackets.\n")
    a("| Recovery | Attack | WER | Speaker cosine (recon vs orig) | Chance cosine | PESQ-wb | STOI |")
    a("|---|---|---|---|---|---|---|")
    for c in configs:
        m = c["metrics"]
        a(f"| `{c['recovery']}` | `{c['attack']}` "
          f"| {_fmt_metric(m['wer'])} "
          f"| {_fmt_metric(m['speaker_cosine'])} "
          f"| {_fmt_metric(m['speaker_cosine_chance'])} "
          f"| {_fmt_metric(m['pesq'])} "
          f"| {_fmt_metric(m['stoi'])} |")
    a("")

    a("## Verdict\n")
    a(_verdict(results))

    a("## Transcript spot-checks\n")
    a("The public snapshot retains the raw JSON metrics and transcript "
      "examples, not reconstructed WAV files. Local evaluation runs may "
      "write sample WAVs under `tests/privacy/reports/samples/`; those "
      "files inherit upstream data/model terms and are ignored by git.\n")
    a("| Configuration | Clip ID | Reference (LibriSpeech) | Whisper-tiny hypothesis |")
    a("|---|---|---|---|")
    for c in configs:
        for ex in c["examples"]:
            ref = ex["reference"][:80].replace("|", "/")
            hyp = ex["hypothesis"][:80].replace("|", "/")
            a(f"| `{c['recovery']}/{c['attack']}` "
              f"| `{ex['clip_id']}` | {ref} | {hyp} |")
    a("")

    if hidden_path:
        lines.extend(_hidden_phrase_section(hidden_path))

    a("")
    a("## How to reproduce\n")
    a("```bash\n"
      "PYTHONPATH=. python tests/privacy/data/download_librispeech.py\n"
      "PYTHONPATH=. python tests/privacy/run_attack_eval.py \\\n"
      "    --corpus tests/privacy/data/librispeech_devclean_3s \\\n"
      "    --out-dir tests/privacy/reports\n"
      "# Hidden-phrase sub-evaluation:\n"
      "PYTHONPATH=. python tests/privacy/data/synthesize_hidden_phrases.py\n"
      "PYTHONPATH=. python tests/privacy/run_attack_eval.py \\\n"
      "    --corpus tests/privacy/data/hidden_phrases \\\n"
      "    --out-dir tests/privacy/reports/hidden_phrases\n"
      "PYTHONPATH=. python tests/privacy/build_report.py\n"
      "```\n")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="tests/privacy/reports/raw_results.json")
    parser.add_argument("--out", default="tests/privacy/REPORT.md")
    parser.add_argument("--hidden",
                        default="tests/privacy/reports/hidden_phrases/raw_results.json",
                        help="optional hidden-phrase raw results JSON")
    args = parser.parse_args()
    hidden = Path(args.hidden) if args.hidden else None
    render(Path(args.raw), Path(args.out), hidden_path=hidden)


if __name__ == "__main__":
    main()
