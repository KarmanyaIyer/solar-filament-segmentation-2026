# SPDX-License-Identifier: AGPL-3.0-only
"""Train YOLO11-seg and dump instance predictions (conf + RLE at cache resolution).

Usage: python yolo_run.py --fold 0 --model yolo11s-seg.pt --epochs 60 --tag y11s
Outputs yolo_pred/<tag>/{val,test}/<stem>.json with [{"conf":..,"rle":..,"box":[..]}].
"""
import argparse
import json
import time
import os
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, folds, load_train  # noqa: E402

CACHE = ROOT / "data" / f"cache{os.environ.get('RES', '1024')}"


def dump(model, stems, split, out, imgsz, conf=0.05):
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    for s in stems:
        img = np.load(CACHE / split.replace("val", "train") / f"{s}.npz")["img"]
        img3 = np.repeat(img[:, :, None], 3, axis=2)
        r = model.predict(img3, imgsz=imgsz, conf=conf, iou=0.6, retina_masks=True, verbose=False, max_det=100)[0]
        items = []
        if r.masks is not None:
            ms = r.masks.data.cpu().numpy() > 0.5
            for m, c, b in zip(ms, r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                items.append({"conf": float(c), "rle": rle["counts"].decode(), "size": rle["size"],
                              "box": [float(v) for v in b]})
        (d / f"{s}.json").write_text(json.dumps(items))


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="y11s")
    ap.add_argument("--ds", default="yolo_ds")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=0, help="stop after the epoch that passes this wall time (0 = off)")
    a = ap.parse_args()
    model = YOLO(a.model)
    if a.max_hours > 0:
        t0 = time.time()

        def _cap(tr):
            if time.time() - t0 > a.max_hours * 3600:
                print(f"time cap {a.max_hours} h reached after epoch {tr.epoch + 1}", flush=True)
                tr.stop = True
        model.add_callback("on_fit_epoch_end", _cap)
    model.train(data=str(Path(a.ds) / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                workers=a.workers, project="yolo_runs", name=a.tag, exist_ok=True, single_cls=True,
                degrees=90, flipud=0.5, fliplr=0.5, mosaic=0.5, scale=0.3, hsv_h=0, hsv_s=0, hsv_v=0.3,
                cos_lr=True, patience=100, plots=False, overlap_mask=False, mask_ratio=2, seed=0)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("best weights:", best, flush=True)
    model = YOLO(str(best))
    out = Path("yolo_pred") / a.tag
    stems = sorted(load_train())
    fm = folds(stems)
    if a.fold >= 0:
        dump(model, [s for s in stems if fm[s] == a.fold], "val", out, a.imgsz)
    dump(model, sorted(p.stem for p in (CACHE / "test").glob("*.npz")), "test", out, a.imgsz)
    print("dumped", out)


if __name__ == "__main__":
    main()
