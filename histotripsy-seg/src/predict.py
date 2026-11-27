"""Run inference and save predicted masks + overlays.
Usage: python -m src.predict --ckpt runs/exp1/best.pt --input dir_or_image.png --out preds/"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model import UNet


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    model = UNet(base=cfg["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg["img_size"]


@torch.no_grad()
def predict_image(model, path, size, device, threshold=0.5):
    img = Image.open(path).convert("L")
    orig_size = img.size
    x = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = torch.from_numpy(x)[None, None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = Image.fromarray(((prob > threshold) * 255).astype(np.uint8)).resize(orig_size, Image.NEAREST)
    return img, mask


def overlay(img, mask, color=(255, 0, 0), alpha=0.4):
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    m = np.asarray(mask) > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(rgb.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_model(a.ckpt, device)
    inp = Path(a.input)
    files = sorted(inp.glob("*.png")) if inp.is_dir() else [inp]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        img, mask = predict_image(model, f, size, device, a.threshold)
        mask.save(out / f"{f.stem}_mask.png")
        overlay(img, mask).save(out / f"{f.stem}_overlay.png")
    print(f"Saved {len(files)} predictions to {out}")


if __name__ == "__main__":
    main()
