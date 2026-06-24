# Preprocessing

Raw B-scans contain machine-generated text overlays (patient identifiers,
timestamps, scan parameters) and UI margins. The released dataset has already
had these removed; this document records the full pipeline for transparency.

## 1. Text / overlay removal (already applied to released images)
Applied to every raw image to produce the released `release-900` set:

1. **Yellow text** &rarr; HSV mask (H 15-40, S>50, V>50, dilate 5px) &rarr;
   Navier-Stokes inpainting.
2. **Orange line/graph** &rarr; HSV mask (H 5-25, S>80, V>80, dilate 3px) &rarr;
   Navier-Stokes inpainting.
3. **Red A-scan graph** &rarr; pasted back from the original (HSV H 0-10 / 160-179,
   S>50, V>50) so the diagnostic waveform is preserved.
4. Resize to 900x900 (LANCZOS).

The released images therefore contain **no burned-in patient identifiers**;
only the ultrasound, the depth axis, and the A-scan waveform remain.

## 2. Margin crop + resize (run locally before training)
`src/preprocess.py` removes residual UI borders with a size-dependent crop
(top 20% / bottom 6% / left 9% / right 20% for <=900px images) and resizes to
the training resolution:

```bash
python src/preprocess.py --src path/to/release-900 --dst data/images --size 512
```

CNNs (EfficientNet-B0, ResNet50, VGG-19-BN) train on these 512x512 images;
ViT-B-16 and the foundation models resize the same images to 224x224 at load
time, so a single preprocessed set serves every model.

> Note: the original training set cropped each image using its *original*
> acquisition resolution. The released images are uniformly 900x900, so
> `preprocess.py` applies the <=900px crop to all of them. This can differ by a
> few pixels from the original crop for the small number of images whose raw
> resolution exceeded 900px, but does not affect the reported trends.
