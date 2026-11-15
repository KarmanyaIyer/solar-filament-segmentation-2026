# SPDX-License-Identifier: AGPL-3.0-only
"""Turn a filament probability map into non-overlapping filament instances.

Steps (all parameters tuned on out-of-fold PQ):
  1. upsample the 1024px probability map to native 2048px (bilinear)
  2. threshold at `thr`
  3. group nearby fragments: components whose pixels lie within `gap` px of each
     other (morphological dilation with a disk of radius gap/2) become one filament
  4. drop instances with area < `min_area` (native px) or mean probability < `min_conf`
The instances are disjoint by construction (labels of one connected-component map).
"""
from __future__ import annotations

import cv2
import numpy as np


def upsample(prob1024: np.ndarray) -> np.ndarray:
    return cv2.resize(prob1024.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)


def instances(prob: np.ndarray, thr=0.5, gap=0, min_area=300, min_conf=0.0, fill_holes=False):
    """prob: (2048,2048) float in [0,1]. Returns list of (mask uint8, score)."""
    fg = (prob > thr).astype(np.uint8)
    if fill_holes:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if fg.sum() == 0:
        return []
    if gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap // 2) + 1, 2 * (gap // 2) + 1))
        grp = cv2.dilate(fg, k)
        n, lab = cv2.connectedComponents(grp, connectivity=8)
        lab = lab * fg
    else:
        n, lab = cv2.connectedComponents(fg, connectivity=8)
    out = []
    if n <= 1:
        return out
    areas = np.bincount(lab.ravel(), minlength=n)
    sums = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n)
    for i in range(1, n):
        if areas[i] < min_area:
            continue
        conf = sums[i] / areas[i]
        if conf < min_conf:
            continue
        out.append(((lab == i).astype(np.uint8), float(conf)))
    return out
