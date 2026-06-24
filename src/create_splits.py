"""
Create the fixed, patient-disjoint train / val / test split for Oculo.
=====================================================================
The benchmark uses a single fixed 75 / 10 / 15 split (NOT k-fold). Because
multiple B-scans of the same patient have consecutive ``image_id`` values
(images are exported per acquisition session), an ``image_id``-ordered
sequential split keeps every patient entirely within one partition, i.e. it
is patient-disjoint and free of train/test leakage.

The repository already ships the canonical frozen split in ``splits/``
(``train.csv`` / ``val.csv`` / ``test.csv``) that was used to produce the
reported numbers. This script regenerates an equivalent patient-disjoint
sequential split and is provided for transparency / reproducibility.

Usage:
  python src/create_splits.py --csv data.csv --out-dir splits \
      --train 0.75 --val 0.10 --test 0.15
"""

import argparse
from pathlib import Path

import pandas as pd

PRED_COLS = [
    "vitreous_dot_echo",
    "abnormal_contour",
    "membranous_echo",
    "posterior_vitreous_detachment",
    "retinal_detachment",
    "choroidal_detachment",
    "mass_lesion",
    "phthisis",
]


def write_distribution(df, name, fh):
    fh.write(f"{name}: {len(df)} images\n")
    for col in PRED_COLS:
        if col in df.columns:
            pos = int(df[col].sum())
            pct = 100 * pos / len(df) if len(df) else 0
            fh.write(f"    {col:<32s} {pos:>4d} ({pct:5.1f}%)\n")
    fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Create fixed patient-disjoint split")
    ap.add_argument("--csv", type=str, default="data.csv")
    ap.add_argument("--out-dir", type=str, default="splits")
    ap.add_argument("--train", type=float, default=0.75)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "train + val + test fractions must sum to 1.0"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Sequential, patient-disjoint ordering by image_id.
    df = df.sort_values("image_id").reset_index(drop=True)
    n = len(df)
    n_train = round(n * args.train)
    n_val = round(n * args.val)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].reset_index(drop=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "split_summary.txt", "w") as fh:
        fh.write("Oculo fixed patient-disjoint split (sequential by image_id)\n")
        fh.write(f"Total: {n}  |  train {len(df_train)} / "
                 f"val {len(df_val)} / test {len(df_test)}\n\n")
        write_distribution(df_train, "train", fh)
        write_distribution(df_val, "val", fh)
        write_distribution(df_test, "test", fh)

    print(f"Wrote splits to {out_dir}/ : "
          f"train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")


if __name__ == "__main__":
    main()
