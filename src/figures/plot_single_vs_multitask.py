"""
Figure 2 - Single-task vs multi-task per-label F1 (EfficientNet-B0 & ViT-B-16).
==============================================================================
Reproduces ``efficientnet_f1_bar.png`` and ``ViT_f1_bar.png`` from the paper.

Per-label F1 values are the test-set F1 per pathology (threshold 0.5) taken
from the corresponding single-task and multi-task runs. After re-running with
``run_seeds.sh`` you can replace the dictionaries below with the seed-averaged
per-label F1 (from ``results/<model>__<tag>/seed_0/test_metrics.json``).

Usage:
  python src/figures/plot_single_vs_multitask.py --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

TASKS = ["VDE", "AC", "ME", "RD", "ML"]
COLOR_SINGLE = "#b2182b"
COLOR_MULTI = "#2166ac"

# Per-label test F1 (single-task vs multi-task).
EFFICIENTNET = {
    "single": [0.9268, 0.8780, 0.8985, 0.9285, 0.9835],
    "multi":  [0.7376, 0.6667, 0.7609, 0.7857, 0.9143],
}
VIT = {
    "single": [0.6571, 0.6364, 0.7708, 0.4000, 0.6207],
    "multi":  [0.7027, 0.8462, 0.7835, 0.6939, 0.9032],
}


def make_plot(spec, ylabel, out_path):
    x = np.arange(len(TASKS))
    bar_w = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    b1 = ax.bar(x - bar_w / 2 - 0.002, spec["single"], bar_w, label="Single-Task",
                color=COLOR_SINGLE, edgecolor="white", linewidth=0.5, zorder=3)
    b2 = ax.bar(x + bar_w / 2 + 0.002, spec["multi"], bar_w, label="Multi-Task",
                color=COLOR_MULTI, edgecolor="white", linewidth=0.5, zorder=3)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.07, h + 0.008, f"{h:.2f}",
                ha="center", va="bottom", fontsize=12, color=COLOR_SINGLE, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2 + 0.07, h + 0.008, f"{h:.2f}",
                ha="center", va="bottom", fontsize=12, color=COLOR_MULTI, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_xlabel("Pathology", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_ylim(0.30, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.with_suffix('.png')} / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_plot(EFFICIENTNET, "F1 Score (EfficientNet-B0)", out / "efficientnet_f1_bar")
    make_plot(VIT, "F1 Score (ViT-B-16)", out / "ViT_f1_bar")


if __name__ == "__main__":
    main()
