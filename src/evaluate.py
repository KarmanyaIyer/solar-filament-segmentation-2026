"""Rubric-style evaluation: PQ, IoU / Dice distributions and 1:n / n:1 relations.

Mirrors the organisers' self-evaluation notebook: ground truth is grouped per
annotator-image, predictions per image, a (GT, prediction) pair is a true positive
when IoU > 0.5. Since our predictions are pixel-disjoint, each prediction can match
at most one GT filament, so this equals one-to-one matching.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from pycocotools import mask as mask_util


def overlap_tables(gt: dict, preds: dict):
    """gt: stem -> {set_id: [rle]}, preds: stem -> [rle]. Returns list of (iou, dice) matrices (n_gt, n_pred)."""
    tabs = []
    for stem, sets in gt.items():
        pr = preds.get(stem, [])
        pa = mask_util.area(pr).astype(np.float64) if pr else np.zeros(0)
        for g in sets.values():
            ga = mask_util.area(g).astype(np.float64) if g else np.zeros(0)
            if len(g) and len(pr):
                iou = np.asarray(mask_util.iou(pr, g, [0] * len(g))).reshape(len(pr), len(g)).T
            else:
                iou = np.zeros((len(g), len(pr)))
            # Dice from IoU and areas: inter = iou * (a + b) / (1 + iou)
            inter = iou * (ga[:, None] + pa[None, :]) / (1 + iou + 1e-12)
            dice = np.where(iou > 0, 2 * inter / (ga[:, None] + pa[None, :] + 1e-12), 0.0)
            tabs.append((iou, dice))
    return tabs


def summarize(tabs, thr=0.5):
    S = TP = FP = FN = 0.0
    gt_deg, pr_deg, ious, dices = [], [], [], []
    for iou, dice in tabs:
        n_gt, n_pr = iou.shape
        if n_gt == 0:
            FP += n_pr; pr_deg += [0] * n_pr; continue
        if n_pr == 0:
            FN += n_gt; gt_deg += [0] * n_gt; continue
        hit = iou > thr
        S += iou[hit].sum(); TP += hit.sum()
        FP += (hit.sum(0) == 0).sum(); FN += (hit.sum(1) == 0).sum()
        pos = iou > 0
        gt_deg += pos.sum(1).tolist(); pr_deg += pos.sum(0).tolist()
        ious += iou[pos].tolist(); dices += dice[pos].tolist()
    den = TP + 0.5 * FP + 0.5 * FN
    return dict(pq=S / den if den else 0.0, sq=S / max(TP, 1), rq=TP / den if den else 0.0,
                tp=int(TP), fp=int(FP), fn=int(FN), ious=np.array(ious), dices=np.array(dices),
                gt_deg=Counter(gt_deg), pred_deg=Counter(pr_deg))


def bootstrap_pq(tabs, n=1000, seed=0, thr=0.5):
    """Bootstrap over annotator-images for a 95% interval of PQ."""
    rng = np.random.default_rng(seed)
    parts = []
    for iou, _ in tabs:
        n_gt, n_pr = iou.shape
        if n_gt == 0 or n_pr == 0:
            parts.append((0.0, 0, n_pr if n_gt == 0 else 0, n_gt if n_pr == 0 else 0)); continue
        hit = iou > thr
        parts.append((iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()))
    parts = np.array(parts, dtype=np.float64)
    vals = []
    for _ in range(n):
        p = parts[rng.integers(0, len(parts), len(parts))].sum(0)
        vals.append(p[0] / (p[1] + 0.5 * p[2] + 0.5 * p[3]))
    return np.percentile(vals, [2.5, 97.5])


def plot_all(summary, title="", path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for a, key, name in [(ax[0], "ious", "IoU"), (ax[1], "dices", "Dice")]:
        v = summary[key]
        a.hist(v, bins=np.linspace(0, 1, 51), weights=np.ones_like(v) / max(len(v), 1), color="#2b4c7e", edgecolor="k", alpha=0.8)
        if len(v):
            a.axvline(v.mean(), color="crimson", ls="--", label=f"mean {v.mean():.3f}")
            a.legend()
        a.set_xlabel(name); a.set_ylabel("fraction of overlapping pairs"); a.set_title(f"{name} distribution (pairs with overlap)")
    g, p = summary["gt_deg"], summary["pred_deg"]
    ks = sorted(set(g) | set(p))
    ax[2].bar([k + 0.2 for k in ks], [g.get(k, 0) for k in ks], 0.4, label="GT filament -> #preds (1:n)", color="#3c8d5a")
    ax[2].bar([k - 0.2 for k in ks], [p.get(k, 0) for k in ks], 0.4, label="prediction -> #GT (n:1)", color="#d07a2d")
    ax[2].set_xlabel("number of overlapping partners"); ax[2].set_ylabel("count"); ax[2].legend(); ax[2].set_title("one-to-many / many-to-one")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
