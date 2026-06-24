# Foundation-model weights

The three foundation models are evaluated from their official pretrained
checkpoints. Weights total ~2.8 GB and are **not** stored in git. Download them
and place them in a local `pretrained/` directory (gitignored).

| Model    | `--model`  | Architecture          | Default weight path                  |
|----------|------------|-----------------------|--------------------------------------|
| USFM     | `usfm`     | BEiT ViT-B/16         | `pretrained/USFM_latest.pth`         |
| VisionFM | `visionfm` | ViT-B/16 (B-US enc.)  | `pretrained/VisionFM_Ultrasound.pth` |
| OpenUS   | `openus`   | VMamba-Small (SSM)    | `pretrained/OpenUS_S.pth`            |

Override a path with `--pretrained-path /path/to/weights.pth`.

## Sources
- **USFM** — Jiao et al., *Medical Image Analysis* 2024. Obtain `USFM_latest.pth`
  from the official USFM release.
- **VisionFM** — Qiu et al., *NEJM AI* 2024. Use the **B-Ultrasound** encoder
  checkpoint.
- **OpenUS** — Zheng et al., 2024. Use the OpenUS-S DINO teacher checkpoint.

## OpenUS extra dependencies
OpenUS uses a VMamba (state-space) backbone and needs the OpenUS source plus
extra packages:

```bash
# clone the OpenUS repo into the project root so that `OpenUS/vmamba_models`
# is importable by src/foundation_models.py
git clone <openus-repo-url> OpenUS
pip install mamba-ssm einops fvcore
```

USFM and VisionFM use standard `timm` ViT backbones and need no extra packages.

> If you only want to reproduce the EfficientNet-B0 / ResNet50 / VGG-19-BN /
> ViT-B-16 results (Table 3), no foundation-model weights are required.
