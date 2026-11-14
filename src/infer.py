"""Model inference helpers: probability maps with flip TTA, OOF generation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, folds  # noqa: E402
from train import build  # noqa: E402

CACHE = ROOT / "data" / f"cache{os.environ.get('RES', '1024')}"


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path, enc, arch="unet"):
    m = build(enc, arch, pretrained=False)
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd)
    return m.to(device()).eval()


@torch.no_grad()
def predict_prob(models, img_u8: np.ndarray, tta=True) -> np.ndarray:
    """img_u8: (R,R) uint8 at the models' training resolution -> (R,R) float32 probability (mean over models and flips)."""
    dev = device()
    x = torch.from_numpy(((img_u8.astype(np.float32) / 255.0 - 0.45) / 0.25)[None, None]).to(dev)
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    acc = torch.zeros_like(x)
    for m in models:
        for f in flips:
            xi = torch.flip(x, f) if f else x
            p = torch.sigmoid(m(xi).float())
            acc += torch.flip(p, f) if f else p
    return (acc / (len(models) * len(flips)))[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir pattern with {fold}, e.g. runs/r34_f{fold}")
    ap.add_argument("--enc", default="resnet34")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--test", action="store_true", help="predict test set with the fold ensemble")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.test:
        # average all fold models on the test set
        ms = [load_model(Path(a.runs.format(fold=f)) / "last.pt", a.enc, a.arch) for f in range(a.nfold)]
        for p in sorted((CACHE / "test").glob("*.npz")):
            prob = predict_prob(ms, np.load(p)["img"])
            np.save(out / f"{p.stem}.npy", np.round(prob * 255).astype(np.uint8))
        print("test done", flush=True)
        return
    stems = sorted(p.stem for p in (CACHE / "train").glob("*.npz"))
    fmap = folds(stems)
    for f in range(a.nfold):
        m = load_model(Path(a.runs.format(fold=f)) / "last.pt", a.enc, a.arch)
        for s in stems:
            if fmap[s] != f:
                continue
            img = np.load(CACHE / "train" / f"{s}.npz")["img"]
            p = predict_prob([m], img)
            np.save(out / f"{s}.npy", np.round(p * 255).astype(np.uint8))
        print("fold", f, "done", flush=True)


if __name__ == "__main__":
    main()
