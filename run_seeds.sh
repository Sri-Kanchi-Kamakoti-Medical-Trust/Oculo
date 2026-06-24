#!/usr/bin/env bash
# ===========================================================================
# Oculo - reproduce all reported benchmark numbers as the mean over 5 seeds
# on the fixed, patient-disjoint split (splits/train.csv, val.csv, test.csv).
#
# Each run is written to:
#     results/<model>__<tag>/seed_<s>/
# Aggregate to mean +/- std afterwards with:
#     python src/aggregate_seeds.py --results-dir results
#
# Prerequisites:
#   - Preprocessed images:  python src/preprocess.py --src <release-900> --dst data/images
#   - Foundation-model weights in pretrained/ (see docs/foundation_models.md)
#     (only needed for the USFM / VisionFM / OpenUS runs)
#
# Tip: run a single model/seed by editing SEEDS / the sections below, or comment
# out blocks you do not need. Single-task trains one binary model per label.
# ===========================================================================
set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4}"
EXCLUDE="pvd,cd,phthisis"          # evaluate the 5 reported labels
ST_LABELS="vde ac me rd ml"
PY="${PY:-python}"

run() {  # run <model> <tag> <extra-args...>
  local model="$1"; local tag="$2"; shift 2
  for s in $SEEDS; do
    echo ">>> ${model} | ${tag} | seed ${s}"
    $PY src/train.py --model "$model" \
        --experiment-tag "${tag}/seed_${s}" \
        --seed "$s" "$@"
  done
}

# ---------------------------------------------------------------------------
# Table 3 - single-task vs multi-task (full fine-tuning), 4 backbones
# ---------------------------------------------------------------------------
for model in efficientnet_b0 resnet50 vgg19 vit_b_16; do
  # Multi-task (5 labels)
  run "$model" "mt_fullft" --exclude-classes "$EXCLUDE"
  # Single-task (one binary model per label)
  for lbl in $ST_LABELS; do
    run "$model" "st_fullft_${lbl}" --single-task "$lbl"
  done
done

# ---------------------------------------------------------------------------
# Table 4 - finetuning strategies (multi-task)
#   Foundation models: Full FT / Linear Probe / LP->FT
#   EfficientNet-B0 baseline: Full FT (above) + LP->FT
# ---------------------------------------------------------------------------
for fm in openus usfm visionfm; do
  run "$fm" "mt_fullft" --exclude-classes "$EXCLUDE"
  run "$fm" "mt_lp"     --exclude-classes "$EXCLUDE" --freeze-backbone
  run "$fm" "mt_lpft"   --exclude-classes "$EXCLUDE" --freeze-backbone --unfreeze-epoch 10
done

run efficientnet_b0 "mt_lpft" --exclude-classes "$EXCLUDE" --freeze-backbone --unfreeze-epoch 10

echo "All runs complete. Aggregating..."
$PY src/aggregate_seeds.py --results-dir results
