"""Create submission.csv for the 180 test images.

Usage (final submission, from precomputed test probability maps and merged detector predictions):
  python make_sub.py --probs test_probs/r34:1 test_probs/effb3:1 test_probs/r34n:1 \
      --yolo-dir yolo_pred/ens_all/test \
      --fuse '{"yconf":0.2,"thr":0.4,"grow":2,"min_area":200,"min_uconf":0.8}' --out submission.csv
Without --yolo-dir it falls back to U-Net-only instances (--thr/--gap/--min-area/--min-conf).
With --weights it runs the U-Nets itself and saves the test probability maps next to the csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, rle_counts  # noqa: E402
from postproc import instances, upsample  # noqa: E402

CACHE = ROOT / "data" / f"cache{os.environ.get('RES', '1024')}"


def validate(df: pd.DataFrame):
    """Fail if any image has overlapping masks, empty masks or duplicate ids."""
    assert df.filament_id.is_unique
    df = df.assign(stem=df.filament_id.str.rsplit("_", n=1).str[0])
    for stem, g in df.groupby("stem"):
        acc = np.zeros((2048, 2048), np.uint8)
        for r in g.segmentation_rle:
            m = mask_util.decode({"size": [2048, 2048], "counts": r.encode("ascii")})
            assert m.sum() > 0, stem
            acc += m
        assert acc.max() <= 1, f"overlap in {stem}"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--enc", default="resnet34")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--probs", nargs="*", default=[], help="precomputed test prob dirs to average (dir:weight)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=500)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--yolo-dir", default="", help="dir of YOLO test <stem>.json -> fuse instead of CC instances")
    ap.add_argument("--fuse", default="{}", help='json kwargs for fuse.fuse, e.g. {"yconf":0.2,"thr":0.4,"grow":4}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pdir = out.with_suffix("")
    pdir.mkdir(exist_ok=True)
    models = []
    if a.weights:
        from infer import load_model, predict_prob
        models = [load_model(w, a.enc, a.arch) for w in a.weights]
    rows = []
    stems = sorted(p.stem for p in (CACHE / "test").glob("*.npz"))
    assert len(stems) == 180
    for s in stems:
        if models:
            p = predict_prob(models, np.load(CACHE / "test" / f"{s}.npz")["img"])
            np.save(pdir / f"{s}.npy", np.round(p * 255).astype(np.uint8))
            prob = upsample(p)
        else:
            tot = 0; prob = 0
            for spec in a.probs:
                d, w = spec.split(":"); w = float(w)
                prob = prob + w * upsample(np.load(Path(d) / f"{s}.npy").astype(np.float32) / 255.0); tot += w
            prob = prob / tot
        if a.yolo_dir:
            from fuse import fuse, load_yolo
            masks = fuse(prob, load_yolo(Path(a.yolo_dir) / f"{s}.json"), **json.loads(a.fuse))
        else:
            masks = [m for m, _ in instances(prob, thr=a.thr, gap=a.gap, min_area=a.min_area, min_conf=a.min_conf)]
        for k, m in enumerate(masks, 1):
            rows.append({"filament_id": f"{s}_{k}", "segmentation_rle": rle_counts(mask_util.encode(np.asfortranarray(m)))})
    df = pd.DataFrame(rows)
    validate(df)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df.filament_id.str.rsplit('_', n=1).str[0].nunique()} images")


if __name__ == "__main__":
    main()
