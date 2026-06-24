"""
Figure 3 - Foundation models (linear probe) vs EfficientNet-B0 (full FT).
========================================================================
Reproduces ``fm_comparison_f1.png`` from the paper: per-label test F1 for
OpenUS, USFM, VisionFM (linear probe on frozen features) compared with the
best task-specific backbone, EfficientNet-B0 (full fine-tuning, multi-task).

Per-label F1 values come from each model's multi-task run
(``results/<model>__mt_lp/.../test_metrics.json`` for the FMs and
``results/efficientnet_b0__mt_fullft/.../test_metrics.json`` for the baseline).

Usage:
  python src/figures/plot_fm_comparison.py --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

TASKS = ["VDE", "AC", "ME", "RD", "ML"]

# Per-label test F1.
MODELS = [
    ("OpenUS",       [0.7273, 0.6000, 0.7451, 0.6667, 0.7273], "#2166ac"),
    ("USFM",         [0.4158, 0.8462, 0.8000, 0.7931, 0.9412], "#7570b3"),
    ("VisionFM",     [0.6400, 0.6364, 0.7778, 0.8136, 0.9412], "#b2182b"),
    ("EfficientNet", [0.7376, 0.6667, 0.7609, 0.7857, 0.9143], "#F0A202"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(TASKS))
    n = len(MODELS)
    bar_w = 0.16
    gap = 0.03
    offsets = np.arange(n) - (n - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (name, vals, color) in enumerate(MODELS):
        pos = x + offsets[idx] * (bar_w + gap)
        bars = ax.bar(pos, vals, bar_w, label=name, color=color,
                      edgecolor="white", linewidth=0.5, zorder=3)
        h_shift = -0.05 if idx < 2 else 0.05
        for i, bar in enumerate(bars):
            h = bar.get_height()
            s = 0 if (i in (0, 1) and idx in (1, 2)) else h_shift
            ax.text(bar.get_x() + bar.get_width() / 2 + s, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=16, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_xlabel("Pathology", fontweight="bold")
    ax.set_ylabel("F1 Score", fontweight="bold")
    ax.set_ylim(0.3, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, framealpha=0.9)
    plt.tight_layout()
    plt.subplots_adjust(top=1.0)
    plt.savefig(out / "fm_comparison_f1.png", dpi=600, bbox_inches="tight")
    plt.savefig(out / "fm_comparison_f1.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'fm_comparison_f1.png'} / .pdf")


if __name__ == "__main__":
    main()
