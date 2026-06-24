"""
Oculo — single training entry point (fixed train/val/test split)
================================================================
Reproduces the benchmark results (Tables 3-4) reported in the Oculo paper.
Uses the fixed, patient-disjoint split in splits/{train,val,test}.csv.
All reported numbers are the mean over seeds 0-4 on this fixed split
(see run_seeds.sh and aggregate_seeds.py).

Multi-task (5 labels):
  python src/train.py --model efficientnet_b0 \
      --experiment-tag mt_fullft --exclude-classes pvd,cd,phthisis \
      --batch-size 8 --epochs 50 --lr 1e-4 --seed 0

Single-task (one label):
  python src/train.py --model efficientnet_b0 \
      --experiment-tag st_fullft_vde --single-task vde --seed 0

Linear probe / LP->FT (foundation models):
  python src/train.py --model usfm --experiment-tag mt_lp \
      --exclude-classes pvd,cd,phthisis --freeze-backbone --seed 0
  python src/train.py --model usfm --experiment-tag mt_lpft \
      --exclude-classes pvd,cd,phthisis --freeze-backbone --unfreeze-epoch 10 --seed 0

CNNs (EfficientNet-B0, ResNet50, VGG-19-BN) train at 512px; ViT-B-16 and the
foundation models (USFM, VisionFM, OpenUS) auto-override to 224px.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
)
import timm

# Ensure sibling modules are importable when script is called from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import BscanDataset, get_train_transforms, get_val_transforms, PRED_COLS
from losses import get_loss_fn
from foundation_models import is_foundation_model, build_foundation_model, FOUNDATION_REGISTRY

warnings.filterwarnings("ignore")


# ========================== MODEL FACTORY ==========================

# The four image-classification backbones benchmarked in the paper.
# Foundation models (usfm, visionfm, openus) are provided by foundation_models.py.
MODEL_REGISTRY = {
    "efficientnet_b0": "efficientnet_b0",
    "vgg19":           "vgg19_bn",
    "resnet50":        "resnet50",
    "vit_b_16":        "vit_base_patch16_224",
}


def create_model(name, num_classes=8, pretrained=True):
    """Create a timm model with custom classifier head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model, model_name):
    """Freeze all backbone parameters, leaving only the classification head trainable."""
    for param in model.parameters():
        param.requires_grad = False

    head_names = ["head", "classifier", "fc", "pre_logits"]
    unfrozen = 0
    for name, param in model.named_parameters():
        for head_name in head_names:
            if head_name in name:
                param.requires_grad = True
                unfrozen += 1
                break

    total = sum(1 for _ in model.parameters())
    frozen = total - unfrozen
    print(f"  [freeze_backbone] Frozen {frozen}/{total} param groups, "
          f"unfrozen {unfrozen} (head only)")
    if unfrozen == 0:
        print("  WARNING: No head parameters found to unfreeze! "
              "Unfreezing last layer as fallback...")
        params = list(model.named_parameters())
        for name, param in reversed(params):
            param.requires_grad = True
            unfrozen += 1
            if unfrozen >= 2:
                break


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ========================== TRAINING ==========================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    running_loss = 0.0
    n_batches = 0
    for images, labels, img_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_ids.extend(img_ids.tolist() if isinstance(img_ids, torch.Tensor) else img_ids)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return running_loss / max(n_batches, 1), all_logits, all_labels, all_ids


def find_optimal_thresholds(logits, labels, label_names):
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    thresholds = {}
    for i, name in enumerate(label_names):
        best_f1, best_t = 0.0, 0.5
        if int(labels_np[:, i].sum()) == 0:
            thresholds[name] = 0.5
            continue
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels_np[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[name] = round(float(best_t), 2)
    return thresholds


def compute_metrics(logits, labels, label_names, thresholds=None):
    """Compute per-class and macro metrics: Acc, Prec, Rec, Spec, F1, AUROC, AP."""
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    if thresholds is None:
        thresholds = {name: 0.5 for name in label_names}
    threshold_arr = np.array([thresholds[n] for n in label_names])
    preds = (probs >= threshold_arr).astype(int)

    metrics = {}
    for i, name in enumerate(label_names):
        y_true = labels_np[:, i]
        y_pred = preds[:, i]
        pos_count = int(y_true.sum())

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec_val = tp / max(tp + fp, 1)
        rec_val = tp / max(tp + fn, 1)
        spec_val = tn / max(tn + fp, 1)
        f1_val = 2 * prec_val * rec_val / max(prec_val + rec_val, 1e-8)

        try:
            auc = roc_auc_score(y_true, probs[:, i]) if pos_count > 0 else float("nan")
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_true, probs[:, i]) if pos_count > 0 else float("nan")
        except ValueError:
            ap = float("nan")

        metrics[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec_val), 4),
            "recall": round(float(rec_val), 4),
            "specificity": round(float(spec_val), 4),
            "f1": round(float(f1_val), 4),
            "auroc": round(float(auc), 4),
            "ap": round(float(ap), 4),
            "threshold": thresholds[name],
            "support": pos_count,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    valid = [name for name in label_names if metrics[name]["support"] > 0]
    for metric_key in ["accuracy", "precision", "recall", "specificity", "f1", "auroc", "ap"]:
        vals = [metrics[n][metric_key] for n in valid if not np.isnan(metrics[n][metric_key])]
        metrics[f"macro_{metric_key}"] = round(float(np.mean(vals)), 4) if vals else 0.0

    return metrics


def compute_confusion_matrices(logits, labels, label_names, thresholds=None):
    """Compute per-class 2×2 confusion matrices and a combined N×N matrix."""
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    if thresholds is None:
        thresholds = {name: 0.5 for name in label_names}
    threshold_arr = np.array([thresholds[n] for n in label_names])
    preds = (probs >= threshold_arr).astype(int)

    per_class_cm = {}
    for i, name in enumerate(label_names):
        cm = confusion_matrix(labels_np[:, i], preds[:, i], labels=[0, 1])
        per_class_cm[name] = cm.tolist()

    # Build N×N co-occurrence confusion matrix (predicted vs actual label combos)
    n_classes = len(label_names)
    combined_cm = np.zeros((n_classes, n_classes), dtype=int)
    for sample_idx in range(len(labels_np)):
        true_pos = set(np.where(labels_np[sample_idx] == 1)[0])
        pred_pos = set(np.where(preds[sample_idx] == 1)[0])
        for t in true_pos:
            for p in pred_pos:
                combined_cm[t, p] += 1

    return per_class_cm, combined_cm.tolist()


def print_metrics_table(metrics, label_names):
    print(f"\n  {'Label':<30s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'Spec':>7s} "
          f"{'F1':>7s} {'AUROC':>7s} {'AP':>7s} {'Thr':>5s} {'Sup':>5s}")
    print("  " + "-" * 110)
    for name in label_names:
        m = metrics[name]
        print(f"  {name:<30s} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['specificity']:>7.4f} {m['f1']:>7.4f} {m['auroc']:>7.4f} {m['ap']:>7.4f} "
              f"{m['threshold']:>5.2f} {m['support']:>5d}")
    print("  " + "-" * 110)
    print(f"  {'MACRO':<30s} {metrics['macro_accuracy']:>7.4f} {metrics['macro_precision']:>7.4f} "
          f"{metrics['macro_recall']:>7.4f} {metrics['macro_specificity']:>7.4f} "
          f"{metrics['macro_f1']:>7.4f} {metrics['macro_auroc']:>7.4f} {metrics['macro_ap']:>7.4f}")


# ========================== GRADCAM ==========================

def generate_gradcam(model, dataset, image_ids, label_names, out_dir, device, img_size):
    """Generate Grad-CAM heatmaps for specified image IDs."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    except ImportError:
        print("  WARNING: pytorch-grad-cam not installed. Skipping GradCAM.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    cam_dir = Path(out_dir) / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    # Identify target layer based on model architecture
    target_layer = None
    model_cpu = model

    # Try common architectures
    if hasattr(model, 'features'):  # EfficientNet, VGG
        target_layer = [model.features[-1]]
    elif hasattr(model, 'layer4'):  # ResNet
        target_layer = [model.layer4[-1]]
    elif hasattr(model, 'blocks'):  # ViT, BEiT
        target_layer = [model.blocks[-1].norm1]
    elif hasattr(model, 'backbone'):  # OpenUS wrapper
        if hasattr(model.backbone, 'layers'):
            target_layer = [model.backbone.layers[-1]]
        elif hasattr(model.backbone, 'features'):
            target_layer = [model.backbone.features[-1]]

    if target_layer is None:
        print("  WARNING: Could not identify target layer for GradCAM. Skipping.")
        return

    # Find indices in dataset for requested image_ids
    all_ids = dataset.df["image_id"].tolist()
    id_to_idx = {int(iid): idx for idx, iid in enumerate(all_ids)}

    # Use reshape_transform for ViT-like models (output has token dimension)
    reshape_transform = None
    is_vit_like = hasattr(model, 'blocks') and not hasattr(model, 'features')
    if is_vit_like:
        h = w = img_size // 16  # patch size 16
        def reshape_transform(tensor):
            # tensor: [B, 1+num_patches, C] -> [B, C, H, W]
            result = tensor[:, 1:, :]  # skip CLS
            result = result.reshape(tensor.size(0), h, w, result.size(-1))
            result = result.permute(0, 3, 1, 2)
            return result

    try:
        cam = GradCAM(model=model, target_layers=target_layer,
                       reshape_transform=reshape_transform)
    except Exception as e:
        print(f"  WARNING: GradCAM init failed: {e}. Skipping.")
        return

    denorm = transforms_module_denormalize()

    for img_id in image_ids:
        if int(img_id) not in id_to_idx:
            print(f"  GradCAM: image_id {img_id} not found in dataset. Skipping.")
            continue

        idx = id_to_idx[int(img_id)]
        image_tensor, labels_tensor, _ = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        # Denormalize for overlay
        rgb_img = denorm(image_tensor).permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        # Generate GradCAM for each active class
        for cls_idx, cls_name in enumerate(label_names):
            short_name = cls_name.replace("vitreous_dot_echo", "VDE") \
                .replace("abnormal_contour", "AC") \
                .replace("membranous_echo", "ME") \
                .replace("retinal_detachment", "RD") \
                .replace("mass_lesion", "ML")

            try:
                targets = [BinaryClassifierOutputTarget(cls_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]  # First image in batch

                overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(rgb_img)
                axes[0].set_title(f"ID {img_id}")
                axes[0].axis("off")
                axes[1].imshow(overlay)
                gt = "+" if labels_tensor[cls_idx] == 1 else "-"
                axes[1].set_title(f"GradCAM: {short_name} (GT: {gt})")
                axes[1].axis("off")
                plt.tight_layout()
                plt.savefig(cam_dir / f"gradcam_{img_id}_{short_name}.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"  GradCAM failed for ID={img_id}, class={cls_name}: {e}")

    print(f"  GradCAM saved to: {cam_dir}/")


def transforms_module_denormalize():
    """Return a function that denormalizes ImageNet-normalized tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def denorm(tensor):
        return tensor * std + mean
    return denorm


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(description="Paper training — clean train/val/test split")
    all_models = list(MODEL_REGISTRY.keys()) + list(FOUNDATION_REGISTRY.keys())
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=all_models)
    parser.add_argument("--img-dir", type=str, default="data/images",
                        help="Directory of preprocessed {image_id}.png files "
                             "(produced by src/preprocess.py).")
    parser.add_argument("--train-csv", type=str, default="splits/train.csv")
    parser.add_argument("--val-csv", type=str, default="splits/val.csv")
    parser.add_argument("--test-csv", type=str, default="splits/test.csv")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal", "asl"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--experiment-tag", type=str, default=None)
    parser.add_argument("--tune-thresholds", action="store_true", default=False)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze backbone, train only classification head (Linear Probe).")
    parser.add_argument("--unfreeze-epoch", type=int, default=0,
                        help="Unfreeze backbone after this many epochs (LP → FT). "
                             "Requires --freeze-backbone.")
    parser.add_argument("--exclude-classes", type=str, default=None)
    parser.add_argument("--single-task", type=str, default=None,
                        help="Train a single binary classifier for ONE class. "
                             "Use short name: vde, ac, me, rd, ml.")
    parser.add_argument("--gradcam-ids", type=str, default=None,
                        help="Comma-separated image IDs for GradCAM generation.")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # ---- Class handling ----
    SHORT_TO_FULL = {
        "vde": "vitreous_dot_echo",
        "ac": "abnormal_contour",
        "me": "membranous_echo",
        "pvd": "posterior_vitreous_detachment",
        "rd": "retinal_detachment",
        "cd": "choroidal_detachment",
        "ml": "mass_lesion",
        "phthisis": "phthisis",
    }

    if args.single_task:
        st = args.single_task.strip().lower()
        if st in SHORT_TO_FULL:
            st_full = SHORT_TO_FULL[st]
        elif st in PRED_COLS:
            st_full = st
        else:
            raise ValueError(f"Unknown single-task class: '{st}'. "
                             f"Valid: {list(SHORT_TO_FULL.keys())}")
        ACTIVE_COLS = [st_full]
        print(f"\n  SINGLE-TASK MODE: {st_full} (binary classification)")
    elif args.exclude_classes:
        exclude_raw = [x.strip().lower() for x in args.exclude_classes.split(",")]
        exclude_full = set()
        for e in exclude_raw:
            if e in SHORT_TO_FULL:
                exclude_full.add(SHORT_TO_FULL[e])
            elif e in PRED_COLS:
                exclude_full.add(e)
            else:
                print(f"  WARNING: Unknown class to exclude: '{e}'")
        ACTIVE_COLS = [c for c in PRED_COLS if c not in exclude_full]
        print(f"\n  Excluding {len(exclude_full)} classes: {sorted(exclude_full)}")
        print(f"  Active classes ({len(ACTIVE_COLS)}): {ACTIVE_COLS}")
    else:
        ACTIVE_COLS = list(PRED_COLS)

    is_fm = is_foundation_model(args.model)
    timm_name = MODEL_REGISTRY.get(args.model)

    # Output directory
    if args.experiment_tag:
        out_dir = Path(args.output_dir) / f"{args.model}__{args.experiment_tag}"
    else:
        out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Determine img_size — ViT / Swin / foundation models need 224
    img_size = args.img_size
    if "vit" in args.model or "swin" in args.model or is_fm:
        img_size = 224
        print(f"  ViT/Swin/Foundation model detected: overriding img_size to {img_size}")

    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    display_name = timm_name if timm_name else args.model.upper()
    strategy = "Full FT"
    if args.freeze_backbone and args.unfreeze_epoch > 0:
        strategy = f"LP→FT (unfreeze@{args.unfreeze_epoch})"
    elif args.freeze_backbone:
        strategy = "Linear Probe (heads-only)"

    print(f"\nModel:      {args.model} ({display_name})")
    print(f"Strategy:   {strategy}")
    print(f"Image dir:  {args.img_dir}")
    print(f"Image size: {img_size}")
    print(f"Loss:       {args.loss} (gamma={args.gamma})")
    print(f"Epochs:     {args.epochs}, BS={args.batch_size}, LR={args.lr}")
    print(f"Patience:   {args.patience}")
    print(f"Classes:    {len(ACTIVE_COLS)}: {[c[:6] for c in ACTIVE_COLS]}")

    # ============ LOAD DATA ============
    print(f"\n  Loading data...")
    print(f"    Train CSV: {args.train_csv}")
    print(f"    Val CSV:   {args.val_csv}")
    print(f"    Test CSV:  {args.test_csv}")

    train_ds = BscanDataset(args.train_csv, args.img_dir, transform=train_tf, label_cols=ACTIVE_COLS)
    val_ds = BscanDataset(args.val_csv, args.img_dir, transform=val_tf, label_cols=ACTIVE_COLS)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ============ MODEL ============
    num_classes = len(ACTIVE_COLS)
    if is_fm:
        model = build_foundation_model(
            args.model, num_classes=num_classes,
            pretrained_path=args.pretrained_path,
        )
    else:
        model = create_model(timm_name, num_classes=num_classes, pretrained=True)

    if args.freeze_backbone:
        freeze_backbone(model, args.model)

    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    pos_weights = train_ds.get_pos_weights()
    criterion = get_loss_fn(args.loss, pos_weights=pos_weights, gamma=args.gamma, device=device)

    # Optimizer + scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ============ TRAINING ============
    print(f"\n{'='*60}")
    print(f"  TRAINING: {args.model} | {strategy}")
    print(f"{'='*60}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    history = []
    epoch_times = []

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at specified epoch
        if args.freeze_backbone and args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"\n  >>> Unfreezing backbone at epoch {epoch} <<<")
            for param in model.parameters():
                param.requires_grad = True
            # Re-create optimizer with all params and lower LR
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-6)
            _, trainable_after = count_parameters(model)
            print(f"  Trainable params after unfreeze: {trainable_after:,}")

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_logits, val_labels, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": round(lr_now, 7),
            "time_s": round(elapsed, 1),
        })

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " *"
        else:
            epochs_no_improve += 1

        print(f"  Epoch {epoch:>2d}/{args.epochs} | "
              f"train={train_loss:.5f} | val={val_loss:.5f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s{improved}")

        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    train_end = time.time()
    training_time_s = train_end - train_start
    epochs_run = len(history)
    avg_epoch_time = np.mean(epoch_times)
    print(f"\n  Training time: {training_time_s:.1f}s ({training_time_s/60:.1f}min)")
    print(f"  Avg epoch: {avg_epoch_time:.1f}s, Best epoch: {best_epoch}")

    # Load best model
    model.load_state_dict(best_state)
    model = model.to(device)

    # ============ VAL EVALUATION ============
    _, val_logits, val_labels, _ = evaluate(model, val_loader, criterion, device)
    thresholds = {name: 0.5 for name in ACTIVE_COLS}
    if args.tune_thresholds:
        thresholds = find_optimal_thresholds(val_logits, val_labels, ACTIVE_COLS)
    val_metrics = compute_metrics(val_logits, val_labels, ACTIVE_COLS, thresholds)
    print(f"\n  Validation metrics ({'tuned' if args.tune_thresholds else 'fixed 0.5'}):")
    print_metrics_table(val_metrics, ACTIVE_COLS)

    # Save model + history
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "best_model.pt")
    with open(model_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

    # ============ TEST EVALUATION ============
    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION")
    print(f"{'='*60}")

    test_ds = BscanDataset(args.test_csv, args.img_dir, transform=val_tf, label_cols=ACTIVE_COLS)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Warmup GPU (1 batch)
    model.eval()
    with torch.no_grad():
        for imgs, _, _ in test_loader:
            _ = model(imgs.to(device))
            break

    # Timed inference
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_start = time.time()
    _, test_logits, test_labels, test_ids = evaluate(model, test_loader, criterion, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_end = time.time()

    n_test = len(test_ids)
    inf_total_s = inf_end - inf_start
    inf_per_sample_ms = (inf_total_s / max(n_test, 1)) * 1000
    print(f"  Inference: {inf_total_s:.3f}s total, {inf_per_sample_ms:.2f}ms/sample ({n_test} samples)")

    test_metrics = compute_metrics(test_logits, test_labels, ACTIVE_COLS, thresholds)
    print(f"\n  Test metrics:")
    print_metrics_table(test_metrics, ACTIVE_COLS)

    # Confusion matrices
    per_class_cm, combined_cm = compute_confusion_matrices(
        test_logits, test_labels, ACTIVE_COLS, thresholds
    )

    # GPU memory
    peak_mem_mb = 0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Save all test results
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)
    with open(out_dir / "test_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(out_dir / "confusion_matrices.json", "w") as f:
        json.dump({
            "per_class": per_class_cm,
            "combined": combined_cm,
            "label_names": ACTIVE_COLS,
        }, f, indent=2)

    # Per-sample predictions
    probs = torch.sigmoid(test_logits).numpy()
    pred_df = pd.DataFrame({"image_id": test_ids})
    for i, name in enumerate(ACTIVE_COLS):
        pred_df[f"{name}_prob"] = probs[:, i].round(4)
        pred_df[f"{name}_pred"] = (probs[:, i] >= thresholds[name]).astype(int)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Save comprehensive summary
    with open(out_dir / "experiment_summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "timm_name": timm_name or args.model,
            "experiment_tag": args.experiment_tag or "default",
            "strategy": strategy,
            "loss": args.loss,
            "gamma": args.gamma,
            "epochs_max": args.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "img_size": img_size,
            "img_dir": args.img_dir,
            "tune_thresholds": args.tune_thresholds,
            "freeze_backbone": args.freeze_backbone,
            "unfreeze_epoch": args.unfreeze_epoch,
            "single_task": args.single_task or "",
            "exclude_classes": args.exclude_classes or "",
            "active_classes": ACTIVE_COLS,
            "num_classes": num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": n_test,
            "timing": {
                "training_total_s": round(training_time_s, 2),
                "training_total_min": round(training_time_s / 60, 2),
                "avg_epoch_s": round(avg_epoch_time, 2),
                "inference_total_s": round(inf_total_s, 3),
                "inference_per_sample_ms": round(inf_per_sample_ms, 2),
                "n_test_samples": n_test,
            },
            "peak_gpu_memory_mb": round(peak_mem_mb, 1),
            "test_macro_metrics": {
                "accuracy": test_metrics["macro_accuracy"],
                "precision": test_metrics["macro_precision"],
                "recall": test_metrics["macro_recall"],
                "specificity": test_metrics["macro_specificity"],
                "f1": test_metrics["macro_f1"],
                "auroc": test_metrics["macro_auroc"],
                "ap": test_metrics["macro_ap"],
            },
            "val_macro_metrics": {
                "accuracy": val_metrics["macro_accuracy"],
                "precision": val_metrics["macro_precision"],
                "recall": val_metrics["macro_recall"],
                "specificity": val_metrics["macro_specificity"],
                "f1": val_metrics["macro_f1"],
                "auroc": val_metrics["macro_auroc"],
                "ap": val_metrics["macro_ap"],
            },
        }, f, indent=2, default=str)

    # ============ GRADCAM ============
    if args.gradcam_ids:
        print(f"\n{'='*60}")
        print(f"  GENERATING GRAD-CAM")
        print(f"{'='*60}")
        gc_ids = [int(x.strip()) for x in args.gradcam_ids.split(",")]
        generate_gradcam(model, test_ds, gc_ids, ACTIVE_COLS, out_dir, device, img_size)

    print(f"\n{'='*60}")
    print(f"  ALL RESULTS SAVED TO: {out_dir}/")
    print(f"{'='*60}")
    print(f"  Test F1={test_metrics['macro_f1']:.4f} | "
          f"AUROC={test_metrics['macro_auroc']:.4f} | "
          f"Train={training_time_s/60:.1f}min | "
          f"Inf={inf_per_sample_ms:.1f}ms/sample")
    print("Done!\n")


if __name__ == "__main__":
    main()
