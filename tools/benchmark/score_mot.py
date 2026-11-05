#!/usr/bin/env python3
"""Score detections / tracks against MOT17 ground truth.

Ground truth: pedestrians (class 1, consider flag 1). MOT17 also labels people that are almost
fully hidden; --min-visibility V turns pedestrians with visibility < V into ignored regions.
Boxes overlapping ignored annotations
(static people, people on vehicles, reflections, distractors, consider=0) by IoU >= 0.5
are dropped before scoring, as in the MOTChallenge evaluation.

  detections:  AP@0.5 (all-point interpolated), plus recall/precision at --conf
  tracks:      MOTA, IDF1, ID switches, FP, FN  (CLEAR-MOT style, IoU >= 0.5)

This is a compact re-implementation for comparing models, not the official devkit.

Usage:
  score_mot.py det  GT_ROOT RESULT_DIR [RESULT_DIR ...]   (RESULT_DIR/<seq>.txt)
  score_mot.py track GT_ROOT RESULT_DIR [RESULT_DIR ...]
"""
import argparse
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

IGNORE_CLASSES = {2, 7, 8, 12}


def load(path, gt=False, min_visibility=0.0):
    frames = defaultdict(list)
    if not os.path.exists(path):
        return frames
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < 6:
            continue
        f, i = int(float(v[0])), int(float(v[1]))
        x, y, w, h = map(float, v[2:6])
        if gt:
            consider, cls, vis = int(v[6]), int(v[7]), float(v[8])
            kind = "gt" if consider == 1 and cls == 1 and vis >= min_visibility else ("ignore" if consider == 0 or cls in IGNORE_CLASSES or cls == 1 else None)
            if kind:
                frames[f].append((kind, i, x, y, x + w, y + h))
        else:
            frames[f].append((i, x, y, x + w, y + h, float(v[6])))
    return frames


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area = lambda t: (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def split_gt(items):
    gt = [(i, b) for k, i, *b in items if k == "gt"]
    ign = [b for k, i, *b in items if k == "ignore"]
    return [i for i, _ in gt], [b for _, b in gt], ign


def drop_ignored(preds, ign, gt_boxes):
    """Remove predictions that match an ignored region better than any real pedestrian."""
    if not preds or not ign:
        return preds
    boxes = [p[1:5] for p in preds]
    to_ign = iou(boxes, ign).max(1)
    to_gt = iou(boxes, gt_boxes).max(1) if gt_boxes else np.zeros(len(preds))
    return [p for p, a, g in zip(preds, to_ign, to_gt) if not (a >= 0.5 and a > g)]


def score_detections(gt, det, conf):
    records, n_gt = [], 0
    for f in sorted(set(gt) | set(det)):
        _, gt_boxes, ign = split_gt(gt.get(f, []))
        n_gt += len(gt_boxes)
        preds = sorted(drop_ignored(det.get(f, []), ign, gt_boxes), key=lambda p: -p[5])
        m = iou([p[1:5] for p in preds], gt_boxes)
        used = set()
        for k, p in enumerate(preds):
            j = -1
            if gt_boxes:
                order = np.argsort(-m[k])
                j = next((int(c) for c in order if m[k, c] >= 0.5 and c not in used), -1)
            if j >= 0:
                used.add(j)
            records.append((p[5], j >= 0))
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    recall = tp / max(1, n_gt)
    precision = tp / np.maximum(1, tp + fp)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    at = [r for r in records if r[0] >= conf]
    tp_c = sum(r[1] for r in at)
    return dict(AP50=ap, recall=tp_c / max(1, n_gt), precision=tp_c / max(1, len(at)), gt=n_gt)


def score_tracks(gt, trk):
    n_gt = fp = fn = idsw = 0
    last = {}                                   # gt id -> track id it was last matched to
    pair_frames = defaultdict(int)              # (gt id, track id) -> frames with IoU >= 0.5
    gt_ids_total, pred_total = 0, 0
    for f in sorted(set(gt) | set(trk)):
        gt_ids, gt_boxes, ign = split_gt(gt.get(f, []))
        preds = drop_ignored(trk.get(f, []), ign, gt_boxes)
        n_gt += len(gt_boxes)
        gt_ids_total += len(gt_boxes)
        pred_total += len(preds)
        m = iou(gt_boxes, [p[1:5] for p in preds])
        pred_ids = [p[0] for p in preds]
        for a in range(len(gt_ids)):
            for b in range(len(pred_ids)):
                if m[a, b] >= 0.5:
                    pair_frames[(gt_ids[a], pred_ids[b])] += 1
        # Prefer keeping last frame's pairing (CLEAR-MOT), then Hungarian on the rest.
        cost = 1 - m
        for a, g in enumerate(gt_ids):
            if g in last and last[g] in pred_ids:
                b = pred_ids.index(last[g])
                if m[a, b] >= 0.5:
                    cost[a, b] -= 10
        cost[m < 0.5] = 1e6
        matches = []
        if gt_boxes and preds:
            rows, cols = linear_sum_assignment(cost)
            matches = [(r, c) for r, c in zip(rows, cols) if m[r, c] >= 0.5]
        for r, c in matches:
            g, t = gt_ids[r], pred_ids[c]
            if g in last and last[g] != t:
                idsw += 1
            last[g] = t
        fp += len(preds) - len(matches)
        fn += len(gt_boxes) - len(matches)
    g_ids = sorted({g for g, _ in pair_frames})
    t_ids = sorted({t for _, t in pair_frames})
    idtp = 0
    if g_ids and t_ids:
        mat = np.zeros((len(g_ids), len(t_ids)))
        gi = {g: k for k, g in enumerate(g_ids)}
        ti = {t: k for k, t in enumerate(t_ids)}
        for (g, t), n in pair_frames.items():
            mat[gi[g], ti[t]] = n
        r, c = linear_sum_assignment(-mat)
        idtp = mat[r, c].sum()
    return dict(MOTA=1 - (fn + fp + idsw) / max(1, n_gt), IDF1=2 * idtp / max(1, gt_ids_total + pred_total),
                IDSW=idsw, FP=fp, FN=fn, gt=n_gt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["det", "track"])
    p.add_argument("gt_root", help="directory holding <seq>/gt/gt.txt")
    p.add_argument("results", nargs="+")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min-visibility", type=float, default=0.0)
    a = p.parse_args()
    for result in a.results:
        files = sorted(f for f in os.listdir(result) if f.endswith(".txt") and f.startswith("MOT"))
        merged = defaultdict(float)
        for f in files:
            seq = f[:-4]
            gt = load(os.path.join(a.gt_root, seq, "gt", "gt.txt"), gt=True, min_visibility=a.min_visibility)
            pred = load(os.path.join(result, f))
            r = score_detections(gt, pred, a.conf) if a.mode == "det" else score_tracks(gt, pred)
            print(f"{os.path.basename(os.path.normpath(result)):16s} {seq:16s} " +
                  " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
            for k, v in r.items():
                merged[k] += v * (r["gt"] if isinstance(v, float) else 1)
        if len(files) > 1:
            print(f"{os.path.basename(os.path.normpath(result)):16s} {'ALL (gt-weighted)':16s} " +
                  " ".join(f"{k}={v / merged['gt']:.3f}" if k not in ("gt", "IDSW", "FP", "FN") else f"{k}={int(v)}"
                           for k, v in merged.items()))


if __name__ == "__main__":
    main()
