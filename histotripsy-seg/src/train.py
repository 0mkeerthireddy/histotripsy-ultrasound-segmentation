import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import UltrasoundSegDataset, list_names, split_names
from src.losses import DiceBCELoss, dice_score, iou_score
from src.model import UNet


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_dice = tot_iou = 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item()
            tot_dice += dice_score(logits, y)
            tot_iou += iou_score(logits, y)
    n = len(loader)
    return tot_loss / n, tot_dice / n, tot_iou / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    names = list_names(cfg["data_dir"])
    train_names, val_names = split_names(names, cfg["val_fraction"], cfg["seed"])
    print(f"{len(train_names)} train / {len(val_names)} val images, device={device}")

    tr = UltrasoundSegDataset(cfg["data_dir"], train_names, cfg["img_size"], augment=True)
    va = UltrasoundSegDataset(cfg["data_dir"], val_names, cfg["img_size"], augment=False)
    tr_dl = DataLoader(tr, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    va_dl = DataLoader(va, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = UNet(base=cfg["base_channels"]).to(device)
    criterion = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best, history = 0.0, []
    for ep in range(1, cfg["epochs"] + 1):
        tl, td, _ = run_epoch(model, tr_dl, device, criterion, opt)
        vl, vd, vi = run_epoch(model, va_dl, device, criterion)
        sched.step()
        history.append(dict(epoch=ep, train_loss=tl, train_dice=td, val_loss=vl, val_dice=vd, val_iou=vi))
        print(f"ep {ep:03d} | train loss {tl:.4f} dice {td:.4f} | val loss {vl:.4f} dice {vd:.4f} iou {vi:.4f}")
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(), "cfg": cfg}, out / "best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=2)
    print(f"Best val Dice: {best:.4f}")


if __name__ == "__main__":
    main()
