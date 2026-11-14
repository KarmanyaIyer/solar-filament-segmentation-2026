"""Fuse YOLO-seg instances (what/where a filament is) with U-Net probabilities (pixel extent).

For each image:
  1. YOLO instances with conf >= `yconf`, sorted by confidence.
  2. Each instance's region = its YOLO mask dilated by `grow` px (native 2048 scale).
  3. Its final mask = region & (U-Net prob > `thr`); if the U-Net finds too little there
     (< `min_frac` of the YOLO mask area) the YOLO mask itself is used (mode-dependent).
  4. Instances are painted in confidence order, so masks are disjoint; remnants smaller
     than `min_area` are dropped.
  5. Optional rescue: U-Net components (postproc.instances) not overlapping any kept
     instance and with mean prob >= `rescue_conf` are added.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import load_train, poly_to_rle  # noqa: E402
from postproc import instances, upsample  # noqa: E402


def load_yolo(path: Path):
    items = json.loads(path.read_text()) if path.exists() else []
    out = []
    for it in items:
        m = mask_util.decode({"size": it["size"], "counts": it["rle"].encode()})
        out.append((it["conf"], m))
    return out


def fill_holes(mm: np.ndarray) -> np.ndarray:
    """Fill background regions fully enclosed by the mask (annotator polygons have no holes)."""
    ys, xs = np.nonzero(mm)
    if len(ys) == 0:
        return mm
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.pad(mm[y0:y1, x0:x1].astype(np.uint8), 1)
    flood = crop.copy()
    cv2.floodFill(flood, np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), np.uint8), (0, 0), 1)
    out = mm.copy()
    out[y0:y1, x0:x1] |= (flood[1:-1, 1:-1] == 0)
    return out


def fuse(prob: np.ndarray, yolo, yconf=0.3, thr=0.3, grow=8, min_area=300, min_frac=0.0,
         rescue_conf=1.1, use_unet=True, min_uconf=0.0, min_peak=0.0, fill=0, close=0):
    """prob: (2048,2048) float; yolo: list of (conf, mask at any res). Returns list of masks."""
    fg = prob > thr
    claimed = np.zeros(prob.shape, bool)
    out = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)) if grow > 0 else None
    for c, m in sorted(yolo, key=lambda t: -t[0]):
        if c < yconf:
            continue
        if m.shape != prob.shape:
            m = cv2.resize(m, prob.shape[::-1], interpolation=cv2.INTER_NEAREST)
        if use_unet:
            reg = cv2.dilate(m, k) if k is not None else m
            mm = (reg > 0) & fg
            if mm.sum() < min_frac * m.sum():
                mm = m > 0
        else:
            mm = m > 0
        if close > 0:
            ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
            mm = cv2.morphologyEx(mm.astype(np.uint8), cv2.MORPH_CLOSE, ck) > 0
        if fill:
            mm = fill_holes(mm)
        mm &= ~claimed
        if mm.sum() < min_area:
            continue
        if min_uconf > 0 and prob[mm].mean() < min_uconf:
            continue
        if min_peak > 0 and prob[mm].max() < min_peak:
            continue
        claimed |= mm
        out.append(mm.astype(np.uint8))
    if rescue_conf <= 1.0:
        for m, conf in instances(prob, thr=thr, gap=12, min_area=400, min_conf=rescue_conf):
            if (m.astype(bool) & claimed).any():
                continue
            claimed |= m.astype(bool)
            out.append(m)
    return out


TR = None


def _init():
    global TR
    cv2.setNumThreads(1)
    TR = load_train()


def eval_image(args):
    stem, oof_dirs, ydir, grid = args
    prob = sum(upsample(np.load(Path(d) / f"{stem}.npy").astype(np.float32) / 255.0) for d in oof_dirs) / len(oof_dirs)
    yolo = load_yolo(Path(ydir) / f"{stem}.json")
    gts = [[poly_to_rle(p) for p in polys] for polys in TR[stem].values()]
    res = []
    for params in grid:
        ms = fuse(prob, yolo, **params)
        pr = [mask_util.encode(np.asfortranarray(m)) for m in ms]
        S = TP = FP = FN = 0
        for g in gts:
            if not pr:
                FN += len(g); continue
            iou = np.asarray(mask_util.iou(pr, g, [0] * len(g))).reshape(len(pr), len(g))
            hit = iou > 0.5
            S += iou[hit].sum(); TP += hit.sum(); FP += (hit.sum(1) == 0).sum(); FN += (hit.sum(0) == 0).sum()
        res.append((S, TP, FP, FN))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--yolo", required=True, help="comma-separated dirs with <stem>.json (val splits of fold models)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--grid", default=None, help='JSON {param: [values]} for the product grid')
    a = ap.parse_args()
    pairs = [(p.stem, yd) for yd in a.yolo.split(",") for p in sorted(Path(yd).glob("*.json"))]
    grid = []
    if a.grid:
        spec = json.loads(a.grid)
        keys = list(spec)
        grid = [dict(zip(keys, v)) for v in itertools.product(*spec.values())]
    else:
        for yconf, thr, grow, ma, mu in itertools.product([0.05, 0.1, 0.15, 0.2, 0.25], [0.35, 0.4, 0.5], [0, 4],
                                                         [300], [0.0, 0.6, 0.7, 0.8]):
            grid.append(dict(yconf=yconf, thr=thr, grow=grow, min_area=ma, min_uconf=mu))
    with Pool(a.procs, initializer=_init) as pool:
        per = pool.map(eval_image, [(s, a.oof.split(","), yd, grid) for s, yd in pairs])
    tot = np.array(per).sum(0)
    rows = []
    for g, (S, TP, FP, FN) in zip(grid, tot):
        pq = S / (TP + 0.5 * FP + 0.5 * FN)
        rows.append((pq, g, int(TP), int(FP), int(FN)))
    for r in sorted(rows, key=lambda r: -r[0])[:15]:
        print(f"{r[0]:.4f} {r[1]} tp {r[2]} fp {r[3]} fn {r[4]}")


if __name__ == "__main__":
    main()
