# SPDX-License-Identifier: AGPL-3.0-only
"""Out-of-fold evaluation of the fused pipeline: PQ / SQ / RQ, bootstrap CI, rubric plots.

Usage:
  python src/eval_oof.py --oof oof/r34,oof/effb3 --yolo yolo_pred/y11s_f0/val,yolo_pred/y11s_f1/val \
      --fuse '{"yconf":0.2,"thr":0.4,"grow":4,"min_area":300,"min_uconf":0.8}' --out report/eval
Each YOLO dir holds predictions of a model that did not see those images; the U-Net OOF dirs hold
out-of-fold probabilities (uint8, 1024px) for every training image. Writes <out>.json and <out>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import gt_rles_for, load_train  # noqa: E402
from evaluate import bootstrap_pq, overlap_tables, plot_all, summarize  # noqa: E402
from fuse import fuse, load_yolo  # noqa: E402
from postproc import instances, upsample  # noqa: E402


def _init():
    cv2.setNumThreads(1)


def predict(args):
    stem, oof_dirs, ydir, params, unet_only = args
    prob = sum(upsample(np.load(Path(d) / f"{stem}.npy").astype(np.float32) / 255.0) for d in oof_dirs) / len(oof_dirs)
    if unet_only:
        ms = [m for m, _ in instances(prob, **params)]
    else:
        ms = fuse(prob, load_yolo(Path(ydir) / f"{stem}.json"), **params)
    return stem, [mask_util.encode(np.asfortranarray(m)) for m in ms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--fuse", required=True, help="fuse() kwargs, or instances() kwargs with --unet-only")
    ap.add_argument("--unet-only", action="store_true", help="U-Net instances only (yolo dirs just select images)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="report/eval")
    a = ap.parse_args()
    params = json.loads(a.fuse)
    jobs = []
    for yd in a.yolo.split(","):
        for p in sorted(Path(yd).glob("*.json")):
            jobs.append((p.stem, a.oof.split(","), yd, params, a.unet_only))
    with Pool(a.procs, initializer=_init) as pool:
        preds = dict(pool.map(predict, jobs))
    tr = load_train()
    gt = gt_rles_for(tr, list(preds))
    tabs = overlap_tables(gt, preds)
    S = summarize(tabs)
    lo, hi = bootstrap_pq(tabs)
    res = {k: S[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}
    res.update(ci=[float(lo), float(hi)], n_images=len(preds), n_annotator_images=len(tabs),
               mean_iou_overlapping=float(S["ious"].mean()), mean_dice_overlapping=float(S["dices"].mean()),
               gt_deg={int(k): int(v) for k, v in S["gt_deg"].items()},
               pred_deg={int(k): int(v) for k, v in S["pred_deg"].items()}, params=params,
               oof=a.oof, yolo=a.yolo)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    if not a.no_plot:
        plot_all(S, f"Out-of-fold evaluation ({len(preds)} images, {len(tabs)} annotator-images): "
                    f"PQ {S['pq']:.3f} [{lo:.3f}, {hi:.3f}]", a.out + ".png")
    print(json.dumps({k: res[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "ci", "n_images")}, default=float))


if __name__ == "__main__":
    main()
