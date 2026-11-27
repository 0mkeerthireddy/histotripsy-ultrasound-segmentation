import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(logits, target, eps=1e-6, threshold=0.5):
    """Hard Dice for evaluation. Inputs: (B,1,H,W)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_score(logits, target, eps=1e-6, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.w, self.eps = bce_weight, eps

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()
        return self.w * bce + (1 - self.w) * dice
