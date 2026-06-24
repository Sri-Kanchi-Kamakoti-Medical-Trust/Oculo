"""
Loss Functions for Multi-Label Classification
==============================================
Implements Focal Loss and Asymmetric Loss for handling class imbalance
in multi-label B-scan classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Works with raw logits (applies sigmoid internally).
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Args:
            gamma:     Focusing parameter (default 2.0).
            alpha:     Per-class weight tensor of shape (num_classes,).
                       If None, no class weighting.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C) raw logits
            targets: (B, C) binary targets
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.unsqueeze(0)  # (1, C)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Ben-Baruch et al., ICCV 2021.

    Applies different focusing to positive vs negative samples:
    - Positive: (1-p)^gamma_pos
    - Negative: p_m^gamma_neg  where p_m = max(p - clip, 0)
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, reduction="mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C) raw logits
            targets: (B, C) binary targets
        """
        probs = torch.sigmoid(logits)

        # Asymmetric clipping for negatives
        probs_neg = (probs - self.clip).clamp(min=0)

        # Separate positive and negative log probabilities
        log_pos = torch.log(probs.clamp(min=1e-8))
        log_neg = torch.log((1 - probs_neg).clamp(min=1e-8))

        loss_pos = -targets * log_pos * ((1 - probs) ** self.gamma_pos)
        loss_neg = -(1 - targets) * log_neg * (probs_neg ** self.gamma_neg)

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(name, pos_weights=None, gamma=2.0, device="cuda"):
    """Factory function to create loss by name."""
    if name == "bce":
        return nn.BCEWithLogitsLoss(
            pos_weight=pos_weights.to(device) if pos_weights is not None else None
        )
    elif name == "focal":
        alpha = pos_weights / pos_weights.sum() * len(pos_weights) if pos_weights is not None else None
        if alpha is not None:
            alpha = alpha.to(device)
        return FocalLoss(gamma=gamma, alpha=alpha).to(device)
    elif name == "asl":
        return AsymmetricLoss().to(device)
    else:
        raise ValueError(f"Unknown loss: {name}")
