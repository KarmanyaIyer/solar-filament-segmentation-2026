"""Train a U-Net against soft consensus masks on the cached images (1024px, or 2048px with RES=2048).

Usage: python train.py --fold 0 --enc resnet34 --epochs 30 --out runs/r34_f0
       RES=2048 python train.py --fold 0 --enc resnet34 --epochs 30 --out runs/r34n_f0
  --fold -1 trains on all observations.
"""
import argparse
import math
import random
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, folds  # noqa: E402

CACHE = ROOT / "data" / f"cache{os.environ.get('RES', '1024')}"


class LazyArr:
    """Per-item loader over cached npz files (keeps RAM flat at native 2048 px resolution)."""

    def __init__(self, stems, key):
        self.stems, self.key = stems, key

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        return np.load(CACHE / "train" / f"{self.stems[i]}.npz")[self.key]


def load_cache(stems):
    if int(os.environ.get("RES", "1024")) >= 1536:
        return LazyArr(stems, "img"), LazyArr(stems, "soft")
    imgs, softs = [], []
    for s in stems:
        z = np.load(CACHE / "train" / f"{s}.npz")
        imgs.append(z["img"]); softs.append(z["soft"])
    return np.stack(imgs), np.stack(softs)


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, softs, crop, train):
        self.imgs, self.softs, self.crop, self.train = imgs, softs, crop, train

    def __len__(self):
        return len(self.imgs) * (4 if self.train else 1)

    def __getitem__(self, i):
        i = i % len(self.imgs)
        img = self.imgs[i].astype(np.float32) / 255.0
        soft = self.softs[i].astype(np.float32) / 255.0
        if self.train:
            c = self.crop
            # bias crops toward filament pixels half the time
            if random.random() < 0.6 and soft.max() > 0:
                ys, xs = np.nonzero(soft > 0.3)
                if len(ys) == 0:
                    ys, xs = np.nonzero(soft > 0)
                k = random.randrange(len(ys))
                y0 = np.clip(ys[k] - random.randint(0, c - 1), 0, img.shape[0] - c)
                x0 = np.clip(xs[k] - random.randint(0, c - 1), 0, img.shape[1] - c)
            else:
                y0 = random.randint(128, img.shape[0] - c - 128)
                x0 = random.randint(128, img.shape[1] - c - 128)
            img = img[y0:y0 + c, x0:x0 + c]; soft = soft[y0:y0 + c, x0:x0 + c]
            if random.random() < 0.5:
                img = img[:, ::-1]; soft = soft[:, ::-1]
            if random.random() < 0.5:
                img = img[::-1]; soft = soft[::-1]
            k = random.randint(0, 3)
            img = np.rot90(img, k); soft = np.rot90(soft, k)
            # photometric: contrast / gamma / brightness
            g = math.exp(random.uniform(-0.3, 0.3))
            img = np.clip(img, 0, 1) ** g
            img = img * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)
            if random.random() < 0.3:
                img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), random.uniform(0.3, 1.2))
        img = (np.ascontiguousarray(img, dtype=np.float32) - 0.45) / 0.25
        return torch.from_numpy(img[None].copy()), torch.from_numpy(np.ascontiguousarray(soft)[None].copy())


def soft_dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def build(enc, arch="unet", pretrained=True):
    """U-Net (or variant) with 1 input channel; ImageNet encoder weights only needed for training."""
    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN}[arch]
    return cls(encoder_name=enc, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--enc", default="resnet34")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (CACHE / "train").glob("*.npz"))
    fmap = folds(stems)
    tr = [s for s in stems if fmap[s] != a.fold]
    va = [s for s in stems if fmap[s] == a.fold]
    ti, ts = load_cache(tr)
    dl = torch.utils.data.DataLoader(DS(ti, ts, a.crop, True), batch_size=a.bs, shuffle=True,
                                     num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    if va:
        vi, vs = load_cache(va)
        vdl = torch.utils.data.DataLoader(DS(vi, vs, 0, False), batch_size=2, num_workers=0)
    model = build(a.enc, a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lo = model(x)
            lo = lo.float()
            loss = F.binary_cross_entropy_with_logits(lo, y) + soft_dice_loss(lo, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        msg = f"ep {ep} train {tl / len(dl):.4f} {time.time() - t0:.0f}s"
        if va and (ep % 5 == 4 or ep == a.epochs - 1):
            model.eval(); vl = 0
            with torch.no_grad():
                for x, y in vdl:
                    x, y = x.to(dev), y.to(dev)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        lo = model(x)
                    lo = lo.float()
                    vl += (F.binary_cross_entropy_with_logits(lo, y) + soft_dice_loss(lo, y)).item()
            vl /= len(vdl); msg += f" val {vl:.4f}"
        print(msg, flush=True)
        torch.save(model.state_dict(), out / "last.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
