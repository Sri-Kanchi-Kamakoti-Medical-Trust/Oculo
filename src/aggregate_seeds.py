"""
Aggregate per-seed metrics into mean +/- std.
=============================================
All reported Oculo numbers are the mean over seeds 0-4 on the fixed split.
``run_seeds.sh`` writes each run to:

    results/<model>__<tag>/seed_<s>/experiment_summary.json

This script scans those seed sub-directories, aggregates the macro test
metrics across seeds, and writes ``aggregate_metrics.json`` (mean and std)
next to them. It also prints a compact table.

Usage:
  python src/aggregate_seeds.py --results-dir results
  python src/aggregate_seeds.py --results-dir results --experiment efficientnet_b0__mt_fullft
"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

MACRO_KEYS = ["accuracy", "precision", "recall", "specificity", "f1", "auroc", "ap"]


def load_seed_macro(seed_dir: Path):
    summary = seed_dir / "experiment_summary.json"
    metrics = seed_dir / "test_metrics.json"
    if summary.exists():
        with open(summary) as fh:
            data = json.load(fh)
        m = data.get("test_macro_metrics")
        if m:
            return {k: m[k] for k in MACRO_KEYS if k in m}
    if metrics.exists():
        with open(metrics) as fh:
            data = json.load(fh)
        return {k: data.get(f"macro_{k}") for k in MACRO_KEYS
                if data.get(f"macro_{k}") is not None}
    return None


def aggregate_experiment(exp_dir: Path):
    seed_dirs = sorted(d for d in exp_dir.glob("seed_*") if d.is_dir())
    if not seed_dirs:
        return None
    per_seed = []
    for sd in seed_dirs:
        macro = load_seed_macro(sd)
        if macro:
            per_seed.append(macro)
    if not per_seed:
        return None

    agg = {"n_seeds": len(per_seed), "seeds": [sd.name for sd in seed_dirs]}
    for k in MACRO_KEYS:
        vals = [s[k] for s in per_seed if k in s and s[k] is not None]
        if vals:
            agg[k] = {"mean": round(mean(vals), 4),
                      "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0}
    return agg


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-seed metrics (mean +/- std)")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--experiment", type=str, default=None,
                    help="Aggregate only this experiment dir name "
                         "(e.g. efficientnet_b0__mt_fullft). Default: all.")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if args.experiment:
        exp_dirs = [root / args.experiment]
    else:
        exp_dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and any(d.glob("seed_*")))

    if not exp_dirs:
        raise SystemExit(f"No experiments with seed_* sub-dirs found under {root}/")

    print(f"{'experiment':<40s} {'n':>2s} {'F1':>14s} {'AUROC':>14s} {'Acc':>14s}")
    print("-" * 90)
    for exp in exp_dirs:
        agg = aggregate_experiment(exp)
        if not agg:
            continue
        with open(exp / "aggregate_metrics.json", "w") as fh:
            json.dump(agg, fh, indent=2)

        def cell(k):
            if k in agg:
                return f"{agg[k]['mean']:.3f}+/-{agg[k]['std']:.3f}"
            return "-"
        print(f"{exp.name:<40s} {agg['n_seeds']:>2d} "
              f"{cell('f1'):>14s} {cell('auroc'):>14s} {cell('accuracy'):>14s}")

    print("\nWrote aggregate_metrics.json into each experiment directory.")


if __name__ == "__main__":
    main()
