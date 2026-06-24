"""
Foundation model loaders for B-scan classification.

Supported models:
  - USFM:     BEiT-style ViT-B/16, pretrained on 2M ultrasound images
  - VisionFM: DINO-style ViT-B/16, pretrained on 3.4M ophthalmic images (B-ultrasound encoder)
  - OpenUS:   VMamba-Small (state space model), self-supervised on 286K ultrasound images

USFM & VisionFM require 224×224 input and use ImageNet normalisation.
OpenUS requires 224×224 input, CUDA, and the openus_env environment.
"""

import os
import sys
import torch
import torch.nn as nn
import timm
from collections import OrderedDict


# ── USFM ──────────────────────────────────────────────────────

def build_usfm(num_classes: int, pretrained_path: str):
    """
    USFM (Jiao et al., Medical Image Analysis 2024).
    Architecture: BEiT-style ViT-B/16 with shared relative position bias.
    Checkpoint is a flat state_dict (no top-level wrapper key).
    
    Key differences from timm beit_base_patch16_224:
      - USFM: norm.* ↔ timm: fc_norm.*
      - USFM: shared rel_pos_bias.* ↔ timm: per-block blocks.X.attn.relative_position_bias_table
      - USFM: mask_token (skip, not needed for classification)
    """
    model = timm.create_model(
        "beit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )

    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    # Remap keys
    new_sd = OrderedDict()
    for k, v in ckpt.items():
        if k == "mask_token":
            continue
        elif k == "norm.weight":
            new_sd["fc_norm.weight"] = v
        elif k == "norm.bias":
            new_sd["fc_norm.bias"] = v
        elif k.startswith("rel_pos_bias."):
            # Shared relative position bias — broadcast to each block
            if k == "rel_pos_bias.relative_position_bias_table":
                for i in range(12):  # 12 transformer blocks
                    new_sd[f"blocks.{i}.attn.relative_position_bias_table"] = v.clone()
            # Skip relative_position_index (registered buffer, auto-created by timm)
            continue
        else:
            new_sd[k] = v

    msg = model.load_state_dict(new_sd, strict=False)
    n_loaded = len(new_sd) - len(msg.unexpected_keys)
    n_total = len(model.state_dict())
    print(f"[USFM] Loaded {n_loaded}/{n_total} params from {pretrained_path}")
    if msg.missing_keys:
        # Should only be head.weight, head.bias (our new classifier)
        print(f"  Missing ({len(msg.missing_keys)}): {msg.missing_keys[:10]}")
    if msg.unexpected_keys:
        print(f"  Unexpected ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:10]}")

    return model


# ── VisionFM ──────────────────────────────────────────────────

def build_visionfm(num_classes: int, pretrained_path: str):
    """
    VisionFM (Qiu et al., NEJM AI 2024) — B-Ultrasound encoder.
    Architecture: standard ViT-B/16 with absolute position embeddings (DINO/iBOT).
    Checkpoint: ckpt['teacher'] with 'backbone.' prefix on encoder keys.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )

    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    teacher_sd = ckpt["teacher"]

    # Strip 'backbone.' prefix, skip DINO projection head keys
    new_sd = OrderedDict()
    for k, v in teacher_sd.items():
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]
            new_sd[new_key] = v
        # Skip head.* (DINO projection head, not a classification head)

    msg = model.load_state_dict(new_sd, strict=False)
    n_loaded = len(new_sd) - len(msg.unexpected_keys)
    n_total = len(model.state_dict())
    print(f"[VisionFM] Loaded {n_loaded}/{n_total} params from {pretrained_path}")
    if msg.missing_keys:
        print(f"  Missing ({len(msg.missing_keys)}): {msg.missing_keys[:10]}")
    if msg.unexpected_keys:
        print(f"  Unexpected ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:10]}")

    return model


# ── OpenUS ────────────────────────────────────────────────────

class OpenUSClassifier(nn.Module):
    """
    Wrapper around OpenUS VMamba-Small backbone for multi-label classification.

    OpenUS (Zheng et al., 2024) uses a VMamba-Small architecture pretrained via
    self-adaptive masked contrastive learning on 286K ultrasound images.

    The backbone produces [B, 1+HW, 768] where token 0 is the CLS (avgpooled).
    We extract the CLS token and pass it through a linear classification head.
    """

    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # backbone returns [B, 1+HW, C]; token 0 is CLS (avgpooled)
        features = self.backbone(x)    # [B, 1+49, 768]
        cls_token = features[:, 0]     # [B, 768]
        logits = self.head(cls_token)  # [B, num_classes]
        return logits


def build_openus(num_classes: int, pretrained_path: str):
    """
    OpenUS-S (Zheng et al., arXiv 2511.11510).
    Architecture: VMamba-Small with depths=[2,2,15,2], dims=96 (→768 final).
    Checkpoint: DINO-style with 'teacher' key, 'backbone.' prefix on encoder keys.

    Requires: mamba_ssm, fvcore, einops (install in openus_env).
    """
    # Add OpenUS repo to path for vmamba_models imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    openus_path = os.path.join(project_root, "OpenUS")
    if openus_path not in sys.path:
        sys.path.insert(0, openus_path)

    from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2

    # Create backbone (no VMamba base weights needed — OpenUS weights are complete)
    backbone = Backbone_DINOv2_VSSM_2()
    embed_dim = backbone.dims[-1]  # 768

    # Load OpenUS pretrained weights from teacher checkpoint
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    teacher_sd = ckpt["teacher"]

    # Strip 'backbone.' prefix, skip DINO projection head keys
    clean_sd = {}
    for k, v in teacher_sd.items():
        if k.startswith("backbone."):
            clean_sd[k[9:]] = v

    msg = backbone.load_state_dict(clean_sd, strict=False)
    n_loaded = len(clean_sd) - len(msg.unexpected_keys)
    n_total = len(backbone.state_dict())
    print(f"[OpenUS] Loaded {n_loaded}/{n_total} backbone params from {pretrained_path}")
    if msg.missing_keys:
        print(f"  Missing ({len(msg.missing_keys)}): {msg.missing_keys[:10]}")
    if msg.unexpected_keys:
        print(f"  Unexpected ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:10]}")

    # Wrap backbone + classification head
    model = OpenUSClassifier(backbone, embed_dim, num_classes)
    return model


# ── Registry ──────────────────────────────────────────────────

FOUNDATION_REGISTRY = {
    "usfm":     build_usfm,
    "visionfm": build_visionfm,
    "openus":   build_openus,
}

# Default weight paths (relative to project root)
FOUNDATION_WEIGHTS = {
    "usfm":     "pretrained/USFM_latest.pth",
    "visionfm": "pretrained/VisionFM_Ultrasound.pth",
    "openus":   "pretrained/OpenUS_S.pth",
}


def is_foundation_model(model_name: str) -> bool:
    return model_name in FOUNDATION_REGISTRY


def build_foundation_model(model_name: str, num_classes: int,
                           pretrained_path: str = None):
    """Build a foundation model with pretrained weights loaded."""
    if pretrained_path is None:
        pretrained_path = FOUNDATION_WEIGHTS[model_name]
    builder = FOUNDATION_REGISTRY[model_name]
    return builder(num_classes, pretrained_path)
