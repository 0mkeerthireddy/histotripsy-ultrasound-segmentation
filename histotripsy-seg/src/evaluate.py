"""Evaluate a checkpoint on a labeled folder: Dice, IoU, and ablation-area error (pixels)."""
import argparse

import torch
from torch.utils.data import DataLoader

from src.dataset import UltrasoundSegDataset, list_names
from src.losses import dice_score, iou_score
from src.predict import load_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_model(a.ckpt, device)
    ds = UltrasoundSegDataset(a.data_dir, list_names(a.data_dir), size)
    dl = DataLoader(ds, batch_size=8)
    dices, ious, area_err = [], [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        dices.append(dice_score(logits, y))
        ious.append(iou_score(logits, y))
        pred = (torch.sigmoid(logits) > 0.5).float()
        area_err.append((pred.sum(dim=(1, 2, 3)) - y.sum(dim=(1, 2, 3))).abs().mean().item())
    n = len(dl)
    print(f"Dice {sum(dices)/n:.4f} | IoU {sum(ious)/n:.4f} | mean |area error| {sum(area_err)/n:.1f} px")


if __name__ == "__main__":
    main()
