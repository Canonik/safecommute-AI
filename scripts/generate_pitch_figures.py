"""Whimsical, eclectic pitch-deck figures for SafeCommute AI.

Hand-drawn xkcd vibes + warm pastel palette.
Uses only glyphs DejaVu Sans can render (no color emoji).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

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
    # no xkcd wobble here — this panel needs predictable spacing
    models = ["SafeCommute", "YAMNet", "PANNs\nCNN14", "AST"]
    params_m = [1.83, 3.7, 80.0, 86.0]
    latency_ms = [12, 50, 150, 200]
    score = [0.804, 0.306, 0.431, 0.485]
    cols = [C["coral"], C["muted"], C["muted"], C["muted"]]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.18, wspace=0.45,
                        left=0.07, right=0.97)

    # panel 1: params (log)
    ax = axes[0]
    ax.bar(models, params_m, color=cols, width=0.68,
           edgecolor=C["ink"], linewidth=2, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, 400)
    ax.set_ylabel("parameters (M, log)", labelpad=6)
    ax.set_title("tiny model", pad=16)
    _grid(ax); ax.set_facecolor(C["paper"])
    for i, v in enumerate(params_m):
        ax.text(i, v * 1.15, f"{v:g}M", ha="center",
                va="bottom", fontsize=11, fontweight="bold")

    # panel 2: latency (linear)
    ax = axes[1]
    ax.bar(models, latency_ms, color=cols, width=0.68,
           edgecolor=C["ink"], linewidth=2, zorder=3)
    ax.set_ylim(0, 280)
    ax.set_ylabel("CPU latency (ms)", labelpad=6)
    ax.set_title("zippy", pad=16)
    _grid(ax); ax.set_facecolor(C["paper"])
    for i, v in enumerate(latency_ms):
        ax.text(i, v + 8, f"{v} ms", ha="center",
                va="bottom", fontsize=11, fontweight="bold")

    # panel 3: score (labels with metric tag)
    ax = axes[2]
    ax.bar(models, score, color=cols, width=0.68,
           edgecolor=C["ink"], linewidth=2, zorder=3)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score", labelpad=6)
    ax.set_title("still smart", pad=16)
    _grid(ax); ax.set_facecolor(C["paper"])
    tags = ["AUC", "mAP", "mAP", "mAP"]
    for i, (v, tag) in enumerate(zip(score, tags)):
        ax.text(i, v + 0.03, f"{v:.2f} {tag}", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", pad=6, labelsize=10)

    _title(fig, "us vs. the big kids",
           "smaller, faster, still sharp  -  same bat, different league")
    fig.text(0.5, 0.02,
             "note: our AUC is binary; SOTA mAP is on AudioSet's 527-class task.  "
             "this chart is about footprint, not a head-to-head accuracy claim.",
             ha="center", fontsize=9, style="italic", color=C["muted"])
    _save(fig, "perf_vs_sota.png")


# ---------- 2. Per-source breakdown ----------
def per_source_breakdown():
    threats = [("as_yell",      90.6),
               ("as_screaming", 79.1),
               ("yt_scream",    78.2),
               ("as_shout",     64.7)]
    negs = [("yt_metro",    64.9),
            ("as_crowd",    42.1),
            ("as_speech",   28.3),
            ("as_laughter", 17.5)]

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

    # side panel: legend + callout, stacked, well clear of the axes
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

    speech_idx = sources.index("as_speech")
    ax.annotate(
        "fine-tune on 1 h of\nsite ambient noise:\n"
        "speech FP drops to ~4%",
        xy=(28.3, speech_idx),
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
    with plt.xkcd(scale=0.5, length=90, randomness=2):
        models = ["SafeCommute", "YAMNet", "PANNs CNN14", "AST"]
        params_m = [1.83, 3.7, 80.0, 86.0]
        latency_ms = [12, 50, 150, 200]
        score = [0.804, 0.306, 0.431, 0.485]
        cols = [C["coral"], C["teal"], C["mustard"], C["plum"]]

        fig, ax = plt.subplots(figsize=(11, 6.8))
        fig.subplots_adjust(top=0.78, left=0.1, right=0.96, bottom=0.13)

        sizes = [500 + s * 3200 for s in score]
        ax.scatter(params_m, latency_ms, s=sizes, c=cols,
                   alpha=0.82, edgecolors=C["ink"],
                   linewidths=2.2, zorder=3)

        offsets = {
            "SafeCommute":   (-0.5, -7),
            "YAMNet":        (1.4,  20),
            "PANNs CNN14":   (-55,  80),
            "AST":           (30,   100),
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
        ax.set_ylabel("CPU latency (ms, log)")
        ax.grid(True, which="both", color=C["muted"],
                linewidth=0.5, linestyle=(0, (1, 3)), alpha=0.55)
        ax.set_axisbelow(True)

        ax.axhspan(5, 30, color=C["sage"], alpha=0.18, zorder=0)
        ax.text(1.0, 22,
                "the cozy edge corner\n(<30 ms, <10M params)",
                fontsize=10, color=C["sage"],
                style="italic", fontweight="bold")
        ax.annotate("lonely over here\n(in a good way)",
                    xy=(1.83, 12), xytext=(6, 6),
                    fontsize=10, color=C["coral"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->",
                                    color=C["coral"], lw=1.8,
                                    connectionstyle="arc3,rad=0.3"))

    _title(fig, "lower-left is where dreams live",
           "small + fast = edge-deployable  -  bubble size = task score")
    _save(fig, "footprint_bubble.png")


# ---------- 4. Confusion matrix ----------
def confusion_matrix():
    cm = np.array([[0.59, 0.41],
                   [0.18, 0.82]])
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

    _title(fig, "the confusion matrix (tastefully)",
           "AUC 0.80  -  acc 70%  -  that 41% slip is the speech problem")
    _save(fig, "confusion_matrix.png")


# ---------- 5. Gamma ablation ----------
def gamma_ablation():
    with plt.xkcd(scale=0.55, length=90, randomness=2):
        gammas = [0.0, 0.5, 1.0, 2.0, 3.0]
        auc = [0.761, 0.804, 0.812, 0.835, 0.856]
        hn = [52.1, 46.9, 31.4, 9.2, 0.0]

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
        ax1.set_facecolor(C["paper"])

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
        _bubble(ax1, (0.5, 0.804),
                "gamma = 0.5\nthe sweet spot",
                (1.25, 0.76), color=C["plum"], curve=-0.3)

    ax2.annotate("gamma = 3 looks great\nuntil hard negatives\ncollapse to zero",
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
           "higher gamma = higher AUC... until the model forgets how to say 'safe'")
    _save(fig, "gamma_ablation.png")


# ---------- 6. Fine-tune impact ----------
def finetune_impact():
    with plt.xkcd(scale=0.5, length=90, randomness=2):
        cats = ["speech FP", "laughter FP", "crowd FP",
                "metro FP", "threat recall"]
        before = [71.7, 82.5, 57.9, 35.1, 78.2]
        after = [4.2, 6.8, 3.1, 2.4, 86.4]

        x = np.arange(len(cats))
        w = 0.38
        fig, ax = plt.subplots(figsize=(12, 6.4))
        fig.subplots_adjust(top=0.76, left=0.08, right=0.96, bottom=0.22)

        b1 = ax.bar(x - w / 2, before, w, label="before",
                    color=C["coral"], edgecolor=C["ink"],
                    linewidth=1.8, zorder=3)
        b2 = ax.bar(x + w / 2, after, w,
                    label="after 10 min fine-tune",
                    color=C["sage"], edgecolor=C["ink"],
                    linewidth=1.8, zorder=3)
        for bars in (b1, b2):
            for bar in bars:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2,
                        f"{v:.1f}", ha="center", fontsize=10,
                        fontweight="bold")

        ax.set_xticks(x, cats, fontsize=11)
        ax.set_ylabel("rate (%)")
        ax.set_ylim(0, 110)
        _grid(ax)
        ax.set_facecolor(C["paper"])
        ax.legend(loc="upper right")

        for i in range(4):
            ax.annotate("", xy=(i + w / 2, after[i] + 7),
                        xytext=(i - w / 2, before[i] - 3),
                        arrowprops=dict(arrowstyle="->",
                                        color=C["plum"], lw=1.6,
                                        connectionstyle="arc3,rad=-0.25"))

        ax.text(2, -22,
                "FPs down is better  -  recall up is better",
                transform=ax.transData, ha="center", fontsize=10,
                style="italic", color=C["muted"])

    _title(fig, "10 minutes that change everything",
           "per-site fine-tuning: the magic step nobody wanted to admit was needed")
    _save(fig, "finetune_impact.png")


# ---------- 7. Privacy pipeline ----------
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
    perf_vs_sota()
    per_source_breakdown()
    footprint_bubble()
    confusion_matrix()
    gamma_ablation()
    finetune_impact()
    privacy_pipeline()
