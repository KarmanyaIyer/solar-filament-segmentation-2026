"""Cache images and soft consensus masks at RES x RES (env RES, default 1024; 2048 = native).

For every training observation (stem) we store:
  img  : uint8 (RES,RES) grayscale H-alpha image (area-downsampled from 2048 when RES < 2048)
  soft : uint8 (RES,RES) = round(255 * fraction of annotator sets marking the pixel)
  n    : number of annotator sets
Only segmentation polygons are used (no spines, boxes or class labels).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, TRAIN_IMG, TEST_IMG, load_train, poly_to_rle  # noqa: E402

import os
R = int(os.environ.get("RES", "1024"))
OUT = ROOT / "data" / f"cache{R}"


def main():
    (OUT / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "test").mkdir(parents=True, exist_ok=True)
    tr = load_train()
    for i, (stem, sets) in enumerate(sorted(tr.items())):
        img = cv2.imread(str(TRAIN_IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        acc = np.zeros((2048, 2048), np.float32)
        for polys in sets.values():
            if polys:
                m = np.zeros((2048, 2048), np.uint8)
                for p in polys:
                    m |= mask_util.decode(poly_to_rle(p))
                acc += m
        acc /= max(len(sets), 1)
        img_s = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
        soft = cv2.resize(acc, (R, R), interpolation=cv2.INTER_AREA)
        np.savez_compressed(OUT / "train" / f"{stem}.npz", img=img_s,
                            soft=np.round(soft * 255).astype(np.uint8), n=len(sets))
        if i % 100 == 0:
            print(i, flush=True)
    for p in sorted(TEST_IMG.glob("*.jpeg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        np.savez_compressed(OUT / "test" / f"{p.stem}.npz",
                            img=cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
