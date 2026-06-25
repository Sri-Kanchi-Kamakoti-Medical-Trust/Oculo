"""
Preprocess released Oculo images for model training.
=====================================================
The released dataset (Hugging Face: SankaraEyeHospital/Oculo) ships the
anonymized images: machine text/overlay already removed via HSV color
masking + Navier-Stokes inpainting, with the A-scan graph preserved, resized
to 900x900.

This script applies the size-dependent margin crop used in the paper
(removing residual UI borders) and resizes to the training resolution,
producing ``{image_id}.png`` files consumed by ``src/train.py`` (default
``--img-dir data/images``).

CNNs (EfficientNet-B0, ResNet50, VGG-19-BN) train at 512px; ViT-B-16 and the
foundation models resize this same cropped image to 224px at load time, so a
single 512px preprocessed set is sufficient for every model.

Usage:
  python src/preprocess.py --src path/to/images --dst data/images --size 512
"""

import argparse
from pathlib import Path

import cv2

# Margin crop (top%, bottom%, left%, right%) for <=900px images, matching the
# paper's preprocessing. Released images are 900x900, so the <=900 rule applies.
CROP_PCT = (20, 6, 9, 20)


def crop_margins(img):
    h, w = img.shape[:2]
    ct, cb, cl, cr = CROP_PCT
    top = int(h * ct / 100)
    bot = h - int(h * cb / 100)
    left = int(w * cl / 100)
    right = w - int(w * cr / 100)
    return img[top:bot, left:right]


def main():
    ap = argparse.ArgumentParser(description="Crop + resize released images for training")
    ap.add_argument("--src", type=str, required=True,
                    help="Directory of released {image_id}.png images.")
    ap.add_argument("--dst", type=str, default="data/images",
                    help="Output directory for preprocessed images.")
    ap.add_argument("--size", type=int, default=512,
                    help="Output square resolution (default 512).")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.png"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not files:
        raise SystemExit(f"No .png images found in {src}")

    n_done, n_err = 0, 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            n_err += 1
            continue
        cropped = crop_margins(img)
        resized = cv2.resize(cropped, (args.size, args.size),
                             interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(dst / f.name), resized)
        n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(files)}]", flush=True)

    print(f"Done: {n_done} images -> {dst} ({args.size}x{args.size})"
          + (f", {n_err} read errors" if n_err else ""))


if __name__ == "__main__":
    main()
