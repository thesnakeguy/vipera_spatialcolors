from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from excluded import EXCLUDED_IDS

DATA_ROOT = Path("/home/u0158953/data/vipers/train")
IMG_SIZE = 512
# ImageNet stats (what smp's timm-efficientnet-b0 expects).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _letterbox(img: Image.Image, size: int, resample, pad: int = 0) -> Image.Image:
    """Resize keeping aspect ratio, then center-pad to size x size."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), resample)
    canvas = Image.new(img.mode, (size, size), pad)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _rand_scale(img: Image.Image, mask: Image.Image, lo: float, hi: float):
    """Random isotropic zoom, kept at `size` by center crop (zoom-in) or pad."""
    s = float(torch.empty(()).uniform_(lo, hi))
    w, h = img.size
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    mask = mask.resize((nw, nh), Image.NEAREST)
    # center back onto a (w, h) canvas (crop if larger, pad if smaller)
    ci, cm = Image.new(img.mode, (w, h), 0), Image.new(mask.mode, (w, h), 0)
    ci.paste(img, ((w - nw) // 2, (h - nh) // 2))
    cm.paste(mask, ((w - nw) // 2, (h - nh) // 2))
    return ci, cm


def _augment(img: Image.Image, mask: Image.Image, strong: bool = False):
    """Train-time augmentation: synced geometry on (img, mask), photometric on img.

    The geometric transforms are applied to BOTH image and mask (the original
    Keras pipeline rotated the image but not the mask -- a desync bug we avoid).
    torch RNG is used so DataLoader workers are seeded correctly. Rotation fills
    with 0, consistent with the letterbox padding.

    strong=True (the "more adaptations" variant) widens the ranges and adds
    random zoom, hue/saturation jitter, and occasional Gaussian blur.
    """
    rot, b, c = (30, 0.6, 0.6) if strong else (15, 0.2, 0.2)  # half-ranges

    if torch.rand(()) < 0.5:
        img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    if torch.rand(()) < 0.5:
        img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
    if strong:
        img, mask = _rand_scale(img, mask, 0.8, 1.2)
    angle = float(torch.empty(()).uniform_(-rot, rot))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    # photometric: image only (mask must not change)
    img = ImageEnhance.Brightness(img).enhance(float(torch.empty(()).uniform_(1 - b, 1 + b)))
    img = ImageEnhance.Contrast(img).enhance(float(torch.empty(()).uniform_(1 - c, 1 + c)))
    if strong:
        img = ImageEnhance.Color(img).enhance(float(torch.empty(()).uniform_(0.6, 1.4)))
        if torch.rand(()) < 0.3:  # mild hue shift via HSV roll
            h, s, v = img.convert("HSV").split()
            h = h.point(lambda p, d=int(torch.randint(-25, 26, ()).item()): (p + d) % 256)
            img = Image.merge("HSV", (h, s, v)).convert("RGB")
        if torch.rand(()) < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(float(torch.empty(()).uniform_(0.5, 1.5))))
    return img, mask


def _rasterize_mask(polys: list[list[float]], w: int, h: int) -> Image.Image:
    """Fill every polygon as 1 on a (h, w) uint8 canvas."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        if len(poly) >= 6:  # need >=3 points
            draw.polygon(poly, fill=1)
    return mask


def split_image_ids(coco: dict, seed: int = 42) -> dict[str, list[int]]:
    """Deterministic 70/15/15 split over sorted image ids.

    Images in `excluded.EXCLUDED_IDS` (unlabeled snakes -- see excluded.py) are
    dropped here, before the shuffle, so they never enter train/val/test.
    """
    ids = sorted(im["id"] for im in coco["images"] if im["id"] not in EXCLUDED_IDS)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


class VipersSeg(Dataset):
    def __init__(self, root: Path | str, image_ids: list[int], coco: dict,
                 size: int = IMG_SIZE, augment: bool = False, strong: bool = False):
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.strong = strong
        self.images = {im["id"]: im for im in coco["images"]}
        # group polygons by image id (an image may have several annotations)
        self.polys: dict[int, list] = {i: [] for i in self.images}
        for a in coco["annotations"]:
            self.polys[a["image_id"]].extend(a["segmentation"])
        # keep only ids whose file is actually on disk
        self.ids = [i for i in image_ids
                    if (self.root / self.images[i]["file_name"]).exists()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        info = self.images[self.ids[idx]]
        w, h = info["width"], info["height"]

        img = Image.open(self.root / info["file_name"]).convert("RGB")
        mask = _rasterize_mask(self.polys[self.ids[idx]], w, h)

        img = _letterbox(img, self.size, Image.BILINEAR, pad=0)
        mask = _letterbox(mask, self.size, Image.NEAREST, pad=0)

        if self.augment:
            img, mask = _augment(img, mask, strong=self.strong)

        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD  # HWC
        x = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()  # CHW
        y = torch.from_numpy(np.asarray(mask, np.float32))[None]  # 1HW
        return x, y


def make_datasets(root: Path | str = DATA_ROOT, size: int = IMG_SIZE, seed: int = 42,
                  aug: str = "default"):
    """aug: 'default' (flips+rot+bright/contrast) or 'strong' (+zoom/hue/sat/blur)."""
    root = Path(root)
    coco = json.load(open(root / "_annotations.coco.json"))
    splits = split_image_ids(coco, seed)
    strong = aug == "strong"
    # augmentation only on train; val/test stay deterministic
    return {k: VipersSeg(root, ids, coco, size, augment=(k == "train"),
                         strong=(strong and k == "train"))
            for k, ids in splits.items()}


if __name__ == "__main__":
    ds = make_datasets()
    for name, d in ds.items():
        print(f"{name:5s}: {len(d)} samples")
    x, y = ds["train"][0]
    print(f"\nsample 0 | image {tuple(x.shape)} {x.dtype} "
          f"range[{x.min():.2f},{x.max():.2f}]")
    print(f"         | mask  {tuple(y.shape)} {y.dtype} "
          f"vals={sorted(set(y.unique().tolist()))} coverage={y.mean():.3f}")
   
    cov = [ds["train"][i][1].mean().item() for i in range(min(20, len(ds["train"])))]
    print(f"         | mask coverage over 20 samples: "
          f"min={min(cov):.3f} mean={np.mean(cov):.3f} max={max(cov):.3f}")
