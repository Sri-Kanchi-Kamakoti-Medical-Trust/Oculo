# Oculo: A Multilabel Dataset for Identification of Ocular Abnormalities from Ultrasound

Benchmark code for the MICCAI 2026 paper introducing **Oculo**, a publicly
available multi-label B-scan ocular-ultrasound dataset (1,630 images,
five annotated abnormalities) and a benchmark of four deep-learning backbones
and three domain-specific foundation models.

- **Dataset:** https://huggingface.co/datasets/SankaraEyeHospital/Oculo
- **Labels (5 evaluated):** Vitreous Dot Echo (VDE), Membranous Echo (ME),
  Retinal Detachment (RD), Mass Lesion (ML), Abnormal Contour (AC).
  Three additional rare labels (PVD, CD, Phthisis) are released but excluded
  from evaluation.

## Models

| Setting | Models |
|---------|--------|
| Backbones (Table 3, single- & multi-task) | EfficientNet-B0, ResNet50, VGG-19-BN, ViT-B-16 |
| Foundation models (Table 4, Full-FT / Linear-Probe / LP&rarr;FT) | OpenUS, USFM, VisionFM |

CNNs train at 512x512; ViT-B-16 and the foundation models at 224x224.

## Repository layout

```
.
├── data.csv                 # image_id + 8 binary labels + diagnosis (1,630 rows)
├── splits/                  # fixed, patient-disjoint 75/10/15 split
│   ├── train.csv  val.csv  test.csv
├── src/
│   ├── train.py             # single training entry point (all models/settings)
│   ├── dataset.py           # multi-label B-scan dataset + transforms
│   ├── losses.py            # focal / BCE / asymmetric loss
│   ├── foundation_models.py # USFM / VisionFM / OpenUS loaders
│   ├── create_splits.py     # regenerate the patient-disjoint split
│   ├── preprocess.py        # crop + resize released images for training
│   ├── aggregate_seeds.py   # per-seed metrics -> mean +/- std
│   └── figures/             # paper figure scripts (Fig 2, Fig 3)
├── run_seeds.sh             # reproduce every reported number over seeds 0-4
├── docs/                    # preprocessing + foundation-model weight notes
├── requirements.txt  .env.example  .gitignore  LICENSE
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# For GPU training, install a CUDA build of torch/torchvision from pytorch.org.
cp .env.example .env
```

## Data

1. Download the dataset (1,630 `release-900` PNGs + `data.csv` + `splits/`)
   from [Hugging Face](https://huggingface.co/datasets/SankaraEyeHospital/Oculo).
2. Preprocess images for training:

   ```bash
   python src/preprocess.py --src path/to/release-900 --dst data/images --size 512
   ```

`data.csv` and the `splits/` CSVs in this repo match the released dataset.

## Training

Single entry point, `src/train.py`. Examples:

```bash
# Multi-task, full fine-tuning (5 labels)
python src/train.py --model efficientnet_b0 --experiment-tag mt_fullft \
    --exclude-classes pvd,cd,phthisis --seed 0

# Single-task (one binary model per label)
python src/train.py --model resnet50 --experiment-tag st_fullft_rd \
    --single-task rd --seed 0

# Foundation model: linear probe / LP->FT
python src/train.py --model usfm --experiment-tag mt_lp \
    --exclude-classes pvd,cd,phthisis --freeze-backbone --seed 0
python src/train.py --model usfm --experiment-tag mt_lpft \
    --exclude-classes pvd,cd,phthisis --freeze-backbone --unfreeze-epoch 10 --seed 0
```

Each run writes metrics, predictions, confusion matrices and a model checkpoint
to `results/<model>__<tag>/...`.

Training protocol: focal loss (γ=2.0), AdamW (lr=1e-4, weight-decay=1e-4),
cosine schedule, batch size 8, up to 50 epochs with early stopping (patience 7),
fixed decision threshold 0.5. ImageNet initialization for CNNs/ViT; pretrained
weights for the foundation models.

## Reproducing the paper numbers (mean over 5 seeds)

All reported metrics are the **mean over seeds 0-4 on the fixed split** in
`splits/`. The split is held constant; only the random seed (head
initialization, batch ordering, stochastic augmentation) varies across runs.

```bash
bash run_seeds.sh                               # all models, seeds 0-4
python src/aggregate_seeds.py --results-dir results   # mean +/- std per experiment
```

`aggregate_seeds.py` writes `aggregate_metrics.json` into each experiment
directory. Table 3 single-task rows are the mean across the five
`st_fullft_<label>` experiments.

## Figures

```bash
python src/figures/plot_single_vs_multitask.py --out-dir figures   # Fig 2
python src/figures/plot_fm_comparison.py --out-dir figures         # Fig 3
```

## Foundation-model weights

The FM checkpoints (~2.8 GB) are not in git. See
[docs/foundation_models.md](docs/foundation_models.md) for download and setup.
The four backbone results (Table 3) need no external weights.

## License

No license is currently applied to this code (all rights reserved). Please
contact the authors for usage permissions. The dataset is released separately
on Hugging Face under its own license (see the dataset card).

## Citation

```bibtex
@inproceedings{oculo2026,
  title     = {Oculo: A Multilabel Dataset for Identification of Ocular Abnormalities from Ultrasound Images},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```
