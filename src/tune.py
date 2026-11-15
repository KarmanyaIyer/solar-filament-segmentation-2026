# SPDX-License-Identifier: AGPL-3.0-only
"""Tune post-processing on out-of-fold probability maps with the exact PQ rule.

For each (thr, gap) we extract candidate instances once, compute their IoU against
every annotator set, then sweep the cheap filters (min_area, min_conf) in numpy.

Usage: python tune.py --oof oof/r34 [--stems-fold 0]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, folds, load_train, poly_to_rle  # noqa: E402
from postproc import instances, upsample  # noqa: E402

TR = None


def _init():
    global TR
    TR = load_train()


def per_image(args):
    stem, oof_dir, thr, gap, extra = args
    prob = upsample(np.load(Path(oof_dir) / f"{stem}.npy").astype(np.float32) / 255.0)
    if extra.get("blend"):
        for d, w in extra["blend"]:
            prob = prob * (1 - w) + w * upsample(np.load(Path(d) / f"{stem}.npy").astype(np.float32) / 255.0)
    inst = instances(prob, thr=thr, gap=gap, min_area=1, min_conf=0.0)
    rles = [mask_util.encode(np.asfortranarray(m)) for m, _ in inst]
    areas = np.array([m.sum() for m, _ in inst], dtype=np.float64)
    confs = np.array([c for _, c in inst], dtype=np.float64)
    sets = []
    for sid, polys in TR[stem].items():
        g = [poly_to_rle(p) for p in polys]
        if rles and g:
            iou = np.asarray(mask_util.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g))
        else:
            iou = np.zeros((len(rles), len(g)))
        sets.append(iou)
    return stem, areas, confs, sets


def score(tables, min_area, min_conf, thr_iou=0.5):
    S = TP = FP = FN = 0.0
    for areas, confs, sets in tables:
        keep = (areas >= min_area) & (confs >= min_conf)
        for iou in sets:
            n_gt = iou.shape[1]
            sub = iou[keep]
            if sub.shape[0] == 0:
                FN += n_gt
                continue
            hit = sub > thr_iou
            S += sub[hit].sum(); TP += hit.sum()
            FP += (hit.sum(1) == 0).sum(); FN += (hit.sum(0) == 0).sum()
    den = TP + 0.5 * FP + 0.5 * FN
    return S / den if den else 0.0, dict(tp=int(TP), fp=int(FP), fn=int(FN), miou=S / max(TP, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--fold", type=int, default=-1, help="restrict to one fold's stems")
    ap.add_argument("--thrs", default="0.3,0.4,0.5,0.6")
    ap.add_argument("--gaps", default="0,8,16,32")
    ap.add_argument("--blend", default="", help="dir:w,dir:w extra prob maps blended in")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    avail = sorted(p.stem for p in Path(a.oof).glob("*.npy"))
    if a.fold >= 0:
        fm = folds(sorted(p.stem for p in (ROOT / "data" / "cache1024" / "train").glob("*.npz")))
        avail = [s for s in avail if fm[s] == a.fold]
    extra = {}
    if a.blend:
        extra["blend"] = [(x.split(":")[0], float(x.split(":")[1])) for x in a.blend.split(",")]
    print(f"{len(avail)} images", flush=True)
    results = []
    with Pool(a.procs, initializer=_init) as pool:
        for thr, gap in itertools.product([float(x) for x in a.thrs.split(",")], [int(x) for x in a.gaps.split(",")]):
            tabs = pool.map(per_image, [(s, a.oof, thr, gap, extra) for s in avail])
            tables = [(t[1], t[2], t[3]) for t in tabs]
            best = None
            for ma, mc in itertools.product([0, 200, 400, 700, 1000, 1500, 2500], [0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]):
                pq, parts = score(tables, ma, mc)
                results.append(dict(thr=thr, gap=gap, min_area=ma, min_conf=mc, pq=pq, **parts))
                if best is None or pq > best[0]:
                    best = (pq, ma, mc, parts)
            print(f"thr {thr} gap {gap}: best PQ {best[0]:.4f} min_area {best[1]} min_conf {best[2]} {best[3]}", flush=True)
    results.sort(key=lambda r: -r["pq"])
    print("TOP:")
    for r in results[:10]:
        print(r)
    (Path(a.oof) / "tune.json").write_text(json.dumps(results[:50], indent=1))


if __name__ == "__main__":
    main()
