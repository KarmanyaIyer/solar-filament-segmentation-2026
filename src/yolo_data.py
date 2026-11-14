"""Build an Ultralytics YOLO-seg dataset from the 1024px cache.

Every annotator set becomes its own training sample (same image, that annotator's
polygons), so conflicting annotations teach the detector a confidence close to the
annotator agreement rate. Split by observation with the same folds as the U-Net.

Usage: python yolo_data.py --fold 0 --out yolo_ds
  --fold -1: every observation except 30 (picked by a hash of the image name) is used for training;
  those 30 are YOLO's validation set, used to select the best epoch.
"""
import argparse
import zlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, folds, load_train  # noqa: E402

CACHE = ROOT / "data" / f"cache{os.environ.get('RES', '1024')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="yolo_ds")
    a = ap.parse_args()
    out = Path(a.out)
    tr = load_train()
    stems = sorted(tr)
    fm = folds(stems)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    nval = 0
    for s in stems:
        is_val = fm[s] == a.fold if a.fold >= 0 else (nval < 30 and zlib.crc32(s.encode()) % 20 == 0)
        split = "val" if is_val else "train"
        nval += is_val
        img = np.load(CACHE / "train" / f"{s}.npz")["img"]
        R = img.shape[0]
        sets = list(tr[s].items())
        if is_val:
            sets = sets[:1]  # one copy is enough for YOLO's own val metrics
        for sid, polys in sets:
            name = sid.replace("/", "_")
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for p in polys:
                xy = np.array(p[0], dtype=np.float64).reshape(-1, 2) / 2048.0
                xy = np.clip(xy, 0, 1)
                if len(xy) >= 3:
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in xy.ravel()))
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    print("done", out)


if __name__ == "__main__":
    main()
