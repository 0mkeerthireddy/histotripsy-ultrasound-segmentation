"""Generate synthetic B-mode-like images (speckle + hypoechoic ablation zone with
hyperechoic rim) for smoke-testing the pipeline. NOT a substitute for real data.
Usage: python scripts/make_synthetic_data.py --out data/synthetic --n 200"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_sample(size=256, rng=None):
    rng = rng or np.random.default_rng()
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = rng.uniform(0.3, 0.7, 2) * size
    a, b = rng.uniform(0.08, 0.2, 2) * size
    th = rng.uniform(0, np.pi)
    xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    r = (xr / a) ** 2 + (yr / b) ** 2
    mask = r < 1.0

    speckle = rng.rayleigh(0.25, (size, size)).astype(np.float32)
    img = 0.35 + speckle * 0.6
    img[mask] *= 0.45                                          # hypoechoic ablated region
    img += 0.35 * np.exp(-((np.sqrt(r) - 1.05) ** 2) / 0.01)   # bright rim
    img *= np.linspace(1.0, 0.6, size)[:, None]                # depth attenuation
    return (img.clip(0, 1) * 255).astype(np.uint8), (mask * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, m = make_sample(a.size, rng)
        Image.fromarray(img).save(out / "images" / f"{i:04d}.png")
        Image.fromarray(m).save(out / "masks" / f"{i:04d}.png")
    print(f"Wrote {a.n} samples to {out}")


if __name__ == "__main__":
    main()
