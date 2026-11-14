"""Merge instance predictions of several YOLO models (same image) into one list.

Instances from all models are clustered greedily in confidence order (mask IoU > `match`, at most one
instance per model per cluster). A cluster's confidence is sum(conf) / number of models, so an instance
found by one model only is down-weighted; its mask is the per-pixel vote >= `vote` x number of models
(vote 0.5 with two models = union of the members).

Usage: python src/yolo_ens.py --dirs yolo_pred/y11s_f0/val,yolo_pred/y11m_f0/val --out yolo_pred/ens_f0/val
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).parent))
from fuse import load_yolo  # noqa: E402


def merge(views, match=0.5, vote=0.5):
    """views: one list of (conf, mask) per model (or TTA view). Returns the merged list of (conf, mask)."""
    n = len(views)
    dets = sorted(((c, m, vi) for vi, v in enumerate(views) for c, m in v), key=lambda t: -t[0])
    clusters = []  # [anchor_rle, member list [(conf, mask)], views used]
    for c, m, vi in dets:
        rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        best, bi = match, -1
        if clusters:
            ious = np.asarray(mask_util.iou([cl[0] for cl in clusters], [rle], [0])).ravel()
            for i in np.argsort(-ious):
                if ious[i] <= match:
                    break
                if vi not in clusters[i][2]:
                    best, bi = ious[i], i
                    break
        if bi < 0:
            clusters.append([rle, [(c, m)], {vi}])
        else:
            clusters[bi][1].append((c, m)); clusters[bi][2].add(vi)
    out = []
    for _, mem, _ in clusters:
        acc = np.zeros(mem[0][1].shape, np.float32)
        for c, m in mem:
            acc += m
        mask = acc >= vote * n if len(mem) > 1 else mem[0][1]
        if mask.sum() == 0:
            mask = mem[0][1]
        out.append((sum(c for c, _ in mem) / n, mask))
    return out


def one(args):
    stem, dirs, out, match, vote = args
    views = [load_yolo(Path(d) / f"{stem}.json") for d in dirs]
    items = []
    for c, m in merge(views, match=match, vote=vote):
        r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        items.append({"conf": float(c), "size": [int(x) for x in r["size"]], "rle": r["counts"].decode()})
    (Path(out) / f"{stem}.json").write_text(json.dumps(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", type=float, default=0.5)
    ap.add_argument("--vote", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    dirs = a.dirs.split(",")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in Path(dirs[0]).glob("*.json"))
    with Pool(a.procs) as pool:
        pool.map(one, [(s, dirs, a.out, a.match, a.vote) for s in stems])
    print(len(stems), "images ->", a.out)


if __name__ == "__main__":
    main()
