# SPDX-License-Identifier: AGPL-3.0-only
"""Inter-annotator panoptic quality: how well one annotator's filaments match another's on the same image.

Every ordered pair (a, b) of different annotators of a multi-annotated observation is scored like a
prediction (a) against ground truth (b), with the competition's matching rule (IoU > 0.5), and the counts
are pooled over all pairs (micro-average), as for the model.

Usage: python src/inter_annotator.py              # all multi-annotated training observations
       python src/inter_annotator.py --folds 0,1  # only the evaluation folds of the report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from common import folds, load_train, poly_to_rle  # noqa: E402


def pair_counts(pred, gt):
    """(sum IoU of matches, TP, FP, FN) of RLE list `pred` against RLE list `gt`."""
    if not pred or not gt:
        return 0.0, 0, len(pred), len(gt)
    iou = np.asarray(mask_util.iou(pred, gt, [0] * len(gt))).reshape(len(pred), len(gt))
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(1) == 0).sum()), int((hit.sum(0) == 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="", help="comma-separated folds to restrict to (default: all)")
    a = ap.parse_args()
    tr = load_train()
    stems = sorted(tr)
    if a.folds:
        fm = folds(stems)
        keep = {int(f) for f in a.folds.split(",")}
        stems = [s for s in stems if fm[s] in keep]
    tot = np.zeros(4)
    n = 0
    for s in stems:
        sets = [[poly_to_rle(p) for p in polys] for polys in tr[s].values()]
        if len(sets) < 2:
            continue
        n += 1
        for i in range(len(sets)):
            for j in range(len(sets)):
                if i != j:
                    tot += pair_counts(sets[i], sets[j])
    S, tp, fp, fn = tot
    print(f"{n} multi-annotated observations: inter-annotator PQ {S / (tp + 0.5 * fp + 0.5 * fn):.4f} "
          f"(SQ {S / tp:.3f}; TP {int(tp)} FP {int(fp)} FN {int(fn)})")


if __name__ == "__main__":
    main()
