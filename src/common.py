"""Shared data loading, RLE helpers and the official-equivalent Panoptic Quality metric.

The PQ implementation mirrors the host's self-evaluation notebook
(kaggle.com/code/azimahmadzadeh/self-evaluation-notebook):
  * ground truth is grouped per annotator-image ("<annotator>-<image>")
  * predictions are grouped per image id (prefix of filament_id before "_")
  * every (gt, pred) pair with IoU > 0.5 is a TP; FP = preds hitting no GT,
    FN = GTs hit by no pred; PQ = sum(IoU_TP) / (TP + 0.5 FP + 0.5 FN), micro-averaged.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG = DATA / "train" / "train_images"
TEST_IMG = DATA / "test" / "test_images"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
H = W = 2048


def load_train():
    """Return dict: image stem -> list of annotator sets; each set is a list of polygon lists."""
    d = json.loads(ANN.read_text())
    by_set = defaultdict(list)
    for a in d["annotations"]:
        by_set[a["image_id"]].append(a["segmentation"])
    stems = defaultdict(dict)
    for im in d["images"]:
        stem = im["file_name"].rsplit(".", 1)[0]
        stems[stem][im["id"]] = by_set.get(im["id"], [])
    return dict(stems)


def poly_to_rle(seg, h=H, w=W):
    rles = mask_util.frPyObjects(seg, h, w)
    return mask_util.merge(rles) if isinstance(rles, list) else rles


def mask_to_rle(m: np.ndarray):
    r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    return r


def rle_counts(r) -> str:
    c = r["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def pq_counts(gt_rles: list, pred_rles: list, thr: float = 0.5):
    """Return (sum_iou_tp, tp, fp, fn) for one annotator set vs one image's predictions."""
    n_gt, n_pr = len(gt_rles), len(pred_rles)
    if n_gt == 0:
        return 0.0, 0, n_pr, 0
    if n_pr == 0:
        return 0.0, 0, 0, n_gt
    iou = mask_util.iou(pred_rles, gt_rles, [0] * n_gt)  # (n_pred, n_gt)
    iou = np.asarray(iou).reshape(n_pr, n_gt).T  # (n_gt, n_pred)
    hit = iou > thr
    s = float(iou[hit].sum())
    tp = int(hit.sum())
    fp = int((hit.sum(0) == 0).sum())
    fn = int((hit.sum(1) == 0).sum())
    return s, tp, fp, fn


def pq_score(gt: dict, preds: dict, thr: float = 0.5, return_parts=False):
    """gt: stem -> {annotator_set_id: [rle,...]}; preds: stem -> [rle,...]."""
    S = TP = FP = FN = 0
    for stem, sets in gt.items():
        pr = preds.get(stem, [])
        for _, g in sets.items():
            s, tp, fp, fn = pq_counts(g, pr, thr)
            S += s; TP += tp; FP += fp; FN += fn
    den = TP + 0.5 * FP + 0.5 * FN
    pq = S / den if den > 0 else 0.0
    if return_parts:
        return pq, dict(sum_iou=S, tp=TP, fp=FP, fn=FN, mean_tp_iou=S / max(TP, 1))
    return pq


def gt_rles_for(stems_dict: dict, stems):
    out = {}
    for s in stems:
        out[s] = {sid: [poly_to_rle(p) for p in polys] for sid, polys in stems_dict[s].items()}
    return out


def folds(stems, k=5, seed=0):
    """Deterministic group split by physical observation (stem)."""
    rng = np.random.RandomState(seed)
    st = sorted(stems)
    rng.shuffle(st)
    return {s: i % k for i, s in enumerate(st)}
