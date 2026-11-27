"""Dataset expecting:  <data_dir>/images/<name>.png  and  <data_dir>/masks/<name>.png
Masks: nonzero pixels = ablation zone."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UltrasoundSegDataset(Dataset):
    def __init__(self, data_dir, names, img_size=256, augment=False):
        self.img_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.names = list(names)
        self.size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        img = Image.open(self.img_dir / name).convert("L").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(self.mask_dir / name).convert("L").resize((self.size, self.size), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask) > 0).astype(np.float32)
        return img, mask

    def _augment(self, img, mask):
        if random.random() < 0.5:  # horizontal flip (lateral axis)
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.8:  # gain / dynamic-range variation
            img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
        if random.random() < 0.5:  # additive noise
            img = np.clip(img + np.random.normal(0, 0.03, img.shape), 0, 1)
        if random.random() < 0.5:  # small translation
            dx, dy = random.randint(-16, 16), random.randint(-16, 16)
            img = np.roll(img, (dy, dx), axis=(0, 1))
            mask = np.roll(mask, (dy, dx), axis=(0, 1))
        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)

    def __getitem__(self, i):
        img, mask = self._load(self.names[i])
        if self.augment:
            img, mask = self._augment(img, mask)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def list_names(data_dir):
    return sorted(p.name for p in (Path(data_dir) / "images").glob("*.png"))


def split_names(names, val_fraction=0.2, seed=42):
    """NOTE: for real data, split by subject/treatment (not by frame) to avoid leakage."""
    rng = random.Random(seed)
    names = names[:]
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_fraction))
    return names[n_val:], names[:n_val]
