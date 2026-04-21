"""Whimsical, eclectic pitch-deck figures for SafeCommute AI.

Hand-drawn xkcd vibes + warm pastel palette.

Every numeric value in this script is loaded from tests/reports/figures_source.json
(produced by `python tests/verify_performance_claims.py --emit-figures-json`).
ZERO hand-typed measurements — so the plots cannot silently drift away from
the model's real behavior.

If tests/reports/figures_source.json is missing, the script exits with a
clear error telling you to run the verifier first. This is by design: every
figure must trace to a measured number.

Re-generate:
    PYTHONPATH=. python tests/verify_performance_claims.py --emit-figures-json \\
        tests/reports/figures_source.json
    python scripts/generate_pitch_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Load model-derived numbers ──────────────────────────────────────────
FIG_JSON = (Path(__file__).resolve().parent.parent
            / "tests" / "reports" / "figures_source.json")
if not FIG_JSON.exists():
    print(f"ERROR: {FIG_JSON} not found.\n"
          "Regenerate it first with:\n"
          "  PYTHONPATH=. python tests/verify_performance_claims.py "
          "--emit-figures-json tests/reports/figures_source.json",
          file=sys.stderr)
    sys.exit(1)

with open(FIG_JSON) as _f:
    DATA = json.load(_f)

C = {
    "peach":   "#ff8c61",
    "coral":   "#ef6461",
    "mustard": "#f4b942",
    "sage":    "#7bb274",
    "teal":    "#3aa6a0",
    "plum":    "#845ec2",
    "pink":    "#ff9fb2",
    "sky":     "#6ac5d9",
    "cream":   "#fff4e3",
    "ink":     "#2d2a32",
    "muted":   "#8d8a93",
    "paper":   "#fdf6ec",
}

plt.rcParams.update({
    "font.family": ["Comic Neue", "Humor Sans", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": C["ink"],
    "axes.labelcolor": C["ink"],
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "text.color": C["ink"],
    "figure.facecolor": C["paper"],
    "axes.facecolor": C["paper"],
    "savefig.facecolor": C["paper"],
    "legend.frameon": False,
    "legend.fontsize": 10,
    "patch.linewidth": 1.4,
})

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["paper"])
    plt.close(fig)
    print(f"wrote {path}")


def _title(fig, title, subtitle=None):
    fig.text(0.03, 0.985, title, fontsize=16, fontweight="bold",
             color=C["ink"], va="top")
    if subtitle:
        fig.text(0.03, 0.935, subtitle, fontsize=10.5,
                 color=C["muted"], va="top", style="italic")


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=C["muted"], linewidth=0.6,
            linestyle=(0, (1, 3)), alpha=0.55)
    ax.set_axisbelow(True)


def _bubble(ax, xy, text, xytext, color, curve=0.3):
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=10.5, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                        connectionstyle=f"arc3,rad={curve}"),
        bbox=dict(boxstyle="round,pad=0.5", fc=C["cream"],
                  ec=color, lw=1.5),
    )


# ---------- 1. Perf vs SOTA ----------
def perf_vs_sota():
    sota = DATA["sota_table"]
    models = sota["models"]
    params_m = sota["params_m"]
    # Latency is historical (not reproduced per-hardware). Label it so.
    latency_ms = sota["latency_ms_historical"]
    # Score: SafeCommute AUC from current checkpoint; others are literature mAP.
    score = list(sota["score"])
    score[0] = DATA["auc_roc"]
    tags = sota["score_metric"]

    # Break CNN14 → two lines for legibility.
    display_models = [m if m != "PANNs-CNN14" else "PANNs\nCNN14" for m in models]
    cols = [C["coral"], C["muted"], C["muted"], C["muted"]]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.18, wspace=0.45,
                        left=0.07, right=0.97)

    ax = axes[0]
    ax.bar(display_models, params_m, color=cols, width=0.68,
           edgecolor=C["ink"], linewidth=2, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, 400)
    ax.set_ylabel("parameters (M, log)", labelpad=6)
    ax.set_title("tiny model", pad=16)
    _grid(ax)
    for i, v in enumerate(params_m):
        ax.text(i, v * 1.15, f"{v:g}M", ha="center",
                va="bottom", fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.bar(display_models, latency_ms, color=cols, width=0.68,
           edgecolor=C["ink"], linewidth=2, zorder=3)
    ax.set_ylim(0, max(latency_ms) * 1.25)
    ax.set_ylabel("CPU latency (ms, historical)", labelpad=6)
    ax.set_title("zippy", pad=16)
    _grid(ax)
    for i, v in enumerate(latency_ms):
        ax.text(i, v + 8, f"{v} ms", ha="center",
                va="bottom", fontsize=11, fontweight="bold")

    ax = axes[2]
    ax.bar(display_models, score, color=cols, width=0.68,
           edgecolor=C["ink"], linewidth=2, zorder=3)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score", labelpad=6)
    ax.set_title("still smart", pad=16)
    _grid(ax)
    for i, (v, tag) in enumerate(zip(score, tags)):
        ax.text(i, v + 0.03, f"{v:.2f} {tag}", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", pad=6, labelsize=10)

    _title(fig, "us vs. the big kids",
           f"params from papers  |  latency = historical  |  "
           f"our AUC {DATA['auc_roc']:.3f} measured, SOTA mAP from literature")
    fig.text(0.5, 0.02,
             "AUC(binary) not directly comparable to mAP(527-class AudioSet).  "
             "this chart is about footprint, not accuracy head-to-head.",
             ha="center", fontsize=9, style="italic", color=C["muted"])
    _save(fig, "perf_vs_sota.png")


# ---------- 2. Per-source breakdown ----------
def per_source_breakdown():
    threats = [(t["source"], t["accuracy"] * 100)
               for t in DATA["threats"]
               if t["source"] in ("as_yell", "as_screaming",
                                  "yt_scream", "as_shout")]
    negs = [(s["source"], (1.0 - (1.0 - s["accuracy"])) * 100)  # accuracy = 1 - FPR
            for s in DATA["safes"]
            if s["source"] in ("yt_metro", "as_crowd",
                               "as_speech", "as_laughter")]
    # Want safes ordered high→low accuracy so the chart reads top-to-bottom.
    negs = sorted(negs, key=lambda r: -r[1])

    sources = [s for s, _ in threats] + [s for s, _ in negs]
    values = [v for _, v in threats] + [v for _, v in negs]
    cols = [C["sage"]] * len(threats) + [C["coral"]] * len(negs)

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    fig.subplots_adjust(top=0.76, left=0.13, right=0.66, bottom=0.12)

    y = np.arange(len(sources))
    ax.barh(y, values, color=cols, height=0.68,
            edgecolor=C["ink"], linewidth=1.8, zorder=3)
    ax.set_yticks(y, sources, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy (%)", labelpad=6)
    ax.set_xlim(0, 100)
    ax.axvline(50, color=C["muted"], linestyle=":", linewidth=1.2)
    _grid(ax, "x")

    for i, v in enumerate(values):
        ax.text(v + 1.8, i, f"{v:.1f}%", va="center",
                fontsize=10.5, fontweight="bold")

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=C["sage"], edgecolor=C["ink"],
                           linewidth=1.5,
                           label="threats  -  strengths"),
            mpatches.Patch(facecolor=C["coral"], edgecolor=C["ink"],
                           linewidth=1.5,
                           label="hard negatives  -  weak"),
        ],
        loc="upper left", bbox_to_anchor=(1.04, 1.0),
        borderaxespad=0, fontsize=10.5,
    )

    fi = DATA.get("finetune_impact")
    if fi:
        speech_after = fi["categories"][0]["after"]
        callout = (f"fine-tune on {fi.get('held_out_wavs','?')}-clip ambient "
                   f"(site={fi['site']}):\nspeech FP drops to "
                   f"{speech_after*100:.1f}%")
    else:
        callout = ("fine-tune on ~1 h of site ambient:\n"
                   "speech FP drops under 10%\n(measurement pending, Step 6)")

    if "as_speech" in sources:
        speech_idx = sources.index("as_speech")
        ax.annotate(
            callout,
            xy=(values[speech_idx], speech_idx),
            xytext=(1.04, 0.30), textcoords="axes fraction",
            fontsize=11, color=C["plum"], fontweight="bold",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=C["plum"],
                            lw=1.8, connectionstyle="arc3,rad=-0.25"),
            bbox=dict(boxstyle="round,pad=0.6", fc=C["cream"],
                      ec=C["plum"], lw=1.5),
        )

    _title(fig, "where we shine  &  where we squint",
           "yells? easy. speech? that's what fine-tuning is for")
    _save(fig, "per_source_breakdown.png")


# ---------- 3. Footprint bubble ----------
def footprint_bubble():
    sota = DATA["sota_table"]
    with plt.xkcd(scale=0.5, length=90, randomness=2):
        models = sota["models"]
        params_m = sota["params_m"]
        latency_ms = sota["latency_ms_historical"]
        score = list(sota["score"])
        score[0] = DATA["auc_roc"]
        cols = [C["coral"], C["teal"], C["mustard"], C["plum"]]

        fig, ax = plt.subplots(figsize=(11, 6.8))
        fig.subplots_adjust(top=0.78, left=0.1, right=0.96, bottom=0.13)

        sizes = [500 + s * 3200 for s in score]
        ax.scatter(params_m, latency_ms, s=sizes, c=cols,
                   alpha=0.82, edgecolors=C["ink"],
                   linewidths=2.2, zorder=3)

        offsets = {
            models[0]: (-0.5, -7),
            models[1]: (1.4,  20),
            models[2]: (-55,  80),
            models[3]: (30,   100),
        }
        for m, x, y, col in zip(models, params_m, latency_ms, cols):
            dx, dy = offsets[m]
            ax.annotate(m, (x, y), xytext=(x + dx, y + dy),
                        fontsize=11.5, fontweight="bold", color=col)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 200)
        ax.set_ylim(5, 600)
        ax.set_xlabel("parameters (M, log)")
        ax.set_ylabel("CPU latency (ms, log — historical)")
        ax.grid(True, which="both", color=C["muted"],
                linewidth=0.5, linestyle=(0, (1, 3)), alpha=0.55)
        ax.set_axisbelow(True)

        ax.axhspan(5, 30, color=C["sage"], alpha=0.18, zorder=0)
        ax.text(1.0, 22,
                "the cozy edge corner\n(<30 ms, <10M params)",
                fontsize=10, color=C["sage"],
                style="italic", fontweight="bold")
        ax.annotate("lonely over here\n(in a good way)",
                    xy=(params_m[0], latency_ms[0]), xytext=(6, 6),
                    fontsize=10, color=C["coral"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->",
                                    color=C["coral"], lw=1.8,
                                    connectionstyle="arc3,rad=0.3"))

    _title(fig, "lower-left is where dreams live",
           "small + fast = edge-deployable  -  bubble size = task score")
    _save(fig, "footprint_bubble.png")


# ---------- 4. Confusion matrix ----------
def confusion_matrix():
    cm = np.array(DATA["confusion_matrix_normalized_at_0_5"])
    labels = ["safe", "unsafe"]
    tile = np.array([[C["sage"], C["coral"]],
                     [C["coral"], C["sage"]]])
    alpha = np.array([[0.55, 0.75],
                      [0.55, 0.90]])

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    fig.subplots_adjust(top=0.78, left=0.18, right=0.95, bottom=0.13)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")

    for i in range(2):
        for j in range(2):
            rect = FancyBboxPatch(
                (j - 0.44, i - 0.44), 0.88, 0.88,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=tile[i, j], alpha=alpha[i, j],
                edgecolor=C["ink"], linewidth=2, zorder=2,
            )
            ax.add_patch(rect)
            ax.text(j, i, f"{cm[i, j] * 100:.1f}%", ha="center",
                    va="center", fontsize=22, fontweight="bold",
                    color=C["ink"])

    ax.set_xticks([0, 1], labels, fontsize=13)
    ax.set_yticks([0, 1], labels, fontsize=13)
    ax.set_xlabel("predicted", fontweight="bold", fontsize=12)
    ax.set_ylabel("actual", fontweight="bold", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    leak = cm[0, 1] * 100        # FP rate at thr 0.5
    _title(fig, "the confusion matrix (tastefully)",
           f"AUC {DATA['auc_roc']:.3f}  -  acc {DATA['accuracy']*100:.0f}%  -  "
           f"{leak:.1f}% safe→unsafe leakage is the speech problem")
    _save(fig, "confusion_matrix.png")


# ---------- 5. Gamma ablation ----------
def gamma_ablation():
    gh = DATA["gamma_ablation_historical"]
    with plt.xkcd(scale=0.55, length=90, randomness=2):
        gammas = gh["gammas"]
        auc = list(gh["auc"])
        hn = gh["hard_neg_acc"]
        # Replace γ=0.5 AUC with the *measured* current checkpoint value.
        for idx, g in enumerate(gammas):
            if abs(g - 0.5) < 1e-9:
                auc[idx] = DATA["auc_roc"]
                break

        fig, ax1 = plt.subplots(figsize=(11.5, 6.8))
        fig.subplots_adjust(top=0.76, left=0.1, right=0.9, bottom=0.22)

        ax1.plot(gammas, auc, marker="o", markersize=14, linewidth=3,
                 color=C["teal"], label="AUC",
                 markeredgecolor=C["ink"],
                 markeredgewidth=1.8, zorder=3)
        ax1.set_xlabel("focal loss  gamma")
        ax1.set_ylabel("AUC", color=C["teal"])
        ax1.tick_params(axis="y", labelcolor=C["teal"])
        ax1.set_ylim(0.72, 0.92)
        _grid(ax1)

        ax2 = ax1.twinx()
        ax2.spines.top.set_visible(False)
        ax2.plot(gammas, hn, marker="D", markersize=13, linewidth=3,
                 color=C["coral"], label="hard-neg accuracy",
                 markeredgecolor=C["ink"],
                 markeredgewidth=1.8, zorder=3)
        ax2.set_ylabel("hard-neg accuracy (%)", color=C["coral"])
        ax2.tick_params(axis="y", labelcolor=C["coral"])
        ax2.set_ylim(-5, 72)

        ax1.axvspan(0.3, 0.7, color=C["mustard"], alpha=0.28, zorder=0)
        _bubble(ax1, (0.5, DATA["auc_roc"]),
                f"gamma = 0.5\nmeasured {DATA['auc_roc']:.3f}\nthe sweet spot",
                (1.25, 0.76), color=C["plum"], curve=-0.3)

    ax2.annotate("gamma = 3 looks great\nuntil hard negatives\ncollapse to zero\n"
                 "(historical snapshot)",
                 xy=(3.0, 0), xytext=(1.9, 40),
                 fontsize=10.5, color=C["coral"], fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["coral"],
                                 lw=1.8, connectionstyle="arc3,rad=0.25"),
                 bbox=dict(boxstyle="round,pad=0.5", fc=C["cream"],
                           ec=C["coral"], lw=1.4))

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=2)
    _title(fig, "the gamma trap",
           "γ=0.5 measured; other γ points are historical "
           "(log-only, checkpoints not preserved)")
    _save(fig, "gamma_ablation.png")


# ---------- 6. Fine-tune impact ----------
def finetune_impact():
    fi = DATA.get("finetune_impact")
    fpr = {s["source"]: 1.0 - s["accuracy"] for s in DATA["safes"]}
    if fi is None:
        # Pre-Step-6: show the pre-fine-tune bars from Phase A + a hatched
        # "pending" band instead of fake after numbers.
        cats = ["speech FP", "laughter FP", "crowd FP",
                "metro FP", "threat recall"]
        before = [fpr.get("as_speech", 0) * 100,
                  fpr.get("as_laughter", 0) * 100,
                  fpr.get("as_crowd", 0) * 100,
                  fpr.get("yt_metro", 0) * 100,
                  DATA["overall_tpr_at_0_5"] * 100]
        after = [None] * len(cats)
    else:
        # Step 6 done — 3 real measured pairs from phase_b_metro.json plus
        # the 2 universal-FP sources (laughter, crowd) held at Phase A as
        # "unchanged — fine-tune did not add these to its training set".
        speech_b = fi["categories"][0]
        overall_b = fi["categories"][1]
        recall_b = fi["categories"][2]
        cats = ["speech FP",
                f"overall FP ({fi['site']} held-out)",
                "threat recall",
                "laughter FP", "crowd FP"]
        before = [(speech_b["before"] or 0) * 100,
                  (overall_b["before"] or 0) * 100,
                  (recall_b["before"] or 0) * 100,
                  fpr.get("as_laughter", 0) * 100,
                  fpr.get("as_crowd", 0) * 100]
        after = [(speech_b["after"] or 0) * 100,
                 (overall_b["after"] or 0) * 100,
                 (recall_b["after"] or 0) * 100,
                 None, None]  # laughter/crowd not measured post-FT

    x = np.arange(len(cats))
    w = 0.38
    with plt.xkcd(scale=0.5, length=90, randomness=2):
        fig, ax = plt.subplots(figsize=(12, 6.4))
        fig.subplots_adjust(top=0.76, left=0.08, right=0.96, bottom=0.22)

        b1 = ax.bar(x - w / 2, before, w, label="before",
                    color=C["coral"], edgecolor=C["ink"],
                    linewidth=1.8, zorder=3)
        for bar, v in zip(b1, before):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2,
                    f"{v:.1f}", ha="center", fontsize=10,
                    fontweight="bold")

        after_vals = [v if v is not None else 0 for v in after]
        b2 = ax.bar(x + w / 2, after_vals, w, label="after (metro fine-tune)",
                    color=[C["sage"] if v is not None else C["muted"]
                           for v in after],
                    edgecolor=C["ink"], linewidth=1.8, zorder=3,
                    hatch=['' if v is not None else '//' for v in after])
        for bar, v in zip(b2, after):
            if v is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 10,
                        "(not measured)", ha="center", va="bottom",
                        fontsize=9, style="italic", color=C["muted"])
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2,
                        f"{v:.1f}", ha="center", fontsize=10,
                        fontweight="bold")

        ax.set_xticks(x, cats, fontsize=11)
        ax.set_ylabel("rate (%)")
        ax.set_ylim(0, 110)
        _grid(ax)
        ax.legend(loc="upper right")

    _title(fig, "the fine-tune delta",
           (f"measured on held-out site ambient "
            f"({fi['site'] if fi else 'pending Step 6'}, "
            f"{fi['held_out_wavs'] if fi else '?'} clips)"))
    _save(fig, "finetune_impact.png")


# ---------- 7. Privacy pipeline (no measurements — layout only) ----------
def privacy_pipeline():
    fig, ax = plt.subplots(figsize=(14.5, 5))
    fig.subplots_adjust(top=0.74, bottom=0.1, left=0.02, right=0.98)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    stages = [
        ("raw audio\n(RAM only)",       7,  C["coral"],   "sensitive"),
        ("mel-spec\n64 bands",          26, C["mustard"], "lossy"),
        ("PCEN\nspectrogram",           46, C["sage"],    "non-invertible"),
        ("CNN + GRU\ninference",        67, C["teal"],    "on-device"),
        ("alert\nSAFE / AMBER / RED",   88, C["plum"],    "a few bytes"),
    ]

    for label, cx, color, tag in stages:
        shadow = FancyBboxPatch(
            (cx - 7.3, 2.9), 14.6, 3.8,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            facecolor=C["ink"], alpha=0.12, zorder=2,
        )
        ax.add_patch(shadow)
        box = FancyBboxPatch(
            (cx - 7.5, 3.1), 15, 3.8,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            linewidth=2.2, edgecolor=C["ink"],
            facecolor=color, alpha=0.85, zorder=3,
        )
        ax.add_patch(box)
        ax.text(cx, 5.4, label, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=C["ink"])
        ax.text(cx, 2.1, tag, ha="center", va="center",
                fontsize=10, style="italic", color=color,
                fontweight="bold")

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 7.7
        x2 = stages[i + 1][1] - 7.7
        ax.add_patch(FancyArrowPatch(
            (x1, 5), (x2, 5),
            arrowstyle="-|>", mutation_scale=22,
            color=C["ink"], linewidth=1.8,
            connectionstyle="arc3,rad=0.15", zorder=4,
        ))

    ax.axvspan(15.5, 53.5, ymin=0.02, ymax=0.14,
               color=C["mustard"], alpha=0.35)
    ax.text(34.5, 0.75,
            "privacy boundary  -  waveform shredded here",
            ha="center", fontsize=10.5, style="italic",
            color=C["coral"], fontweight="bold")

    _title(fig, "privacy by construction (not by promise)",
           "the waveform literally cannot survive past the PCEN stage")
    _save(fig, "privacy_pipeline.png")


if __name__ == "__main__":
    print(f"Reading measured numbers from {FIG_JSON}")
    print(f"  generated_at: {DATA.get('generated_at', 'unknown')}")
    print(f"  AUC {DATA['auc_roc']:.4f}  acc {DATA['accuracy']:.4f}  "
          f"FPR {DATA['overall_fpr_at_0_5']:.4f}")
    print()
    perf_vs_sota()
    per_source_breakdown()
    footprint_bubble()
    confusion_matrix()
    gamma_ablation()
    finetune_impact()
    privacy_pipeline()
