"""
B-scan Multi-Label Dataset
==========================
PyTorch Dataset for ocular ultrasound B-scan multi-label classification.
Reads image IDs + labels from a CSV, loads PNGs from an image folder.
Applies optional transforms (online augmentation during training).
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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


def get_train_transforms(img_size=512):
    """Online augmentation for training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=512):
    """Deterministic transforms for validation/test."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class BscanDataset(Dataset):
    """Multi-label B-scan classification dataset."""

    def __init__(self, csv_path, img_dir, transform=None, label_cols=None):
        """
        Args:
            csv_path:   Path to CSV with image_id + label columns.
            img_dir:    Directory containing {image_id}.png files.
            transform:  torchvision transforms to apply.
            label_cols: List of label column names (default: PRED_COLS).
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.label_cols = label_cols or PRED_COLS

        # Verify all images exist
        missing = []
        for img_id in self.df["image_id"]:
            if not (self.img_dir / f"{img_id}.png").exists():
                missing.append(img_id)
        if missing:
            print(f"  WARNING: {len(missing)} images missing from {img_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row["image_id"])
        img_path = self.img_dir / f"{img_id}.png"

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Multi-label target vector
        labels = torch.tensor(
            [float(row[col]) for col in self.label_cols],
            dtype=torch.float32,
        )

        return image, labels, img_id

    def get_pos_weights(self):
        """Compute positive class weights for BCEWithLogitsLoss (neg/pos ratio)."""
        counts = self.df[self.label_cols].sum().values
        total = len(self.df)
        neg_counts = total - counts
        weights = neg_counts / (counts + 1e-6)
        return torch.tensor(weights, dtype=torch.float32)

    def get_label_frequencies(self):
        """Return fraction of positive samples per label."""
        counts = self.df[self.label_cols].sum().values
        return counts / len(self.df)
