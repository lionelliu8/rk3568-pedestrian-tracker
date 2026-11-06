#!/usr/bin/env python3
"""Does a tracker keep a pedestrian's ID through an occlusion?

An occlusion event is a run of frames where a ground-truth pedestrian's visibility is below
--hidden (default 0.2), with --context frames of good visibility (>= --visible, default 0.5)
within 30 frames before it and at least one such frame within 30 frames after it. For each event the script takes the track ID matched to the person
(IoU >= 0.5, per-frame Hungarian) just before and just after, and classifies the event:

  kept       the same track ID before and after
  switched   a different track ID after
  missed     no matched track in the first visible frames after reappearing
  (events with no matched track before the occlusion are skipped: nothing to keep)

Usage: occlusion_reid.py GT_ROOT RESULT_DIR [RESULT_DIR ...]   (RESULT_DIR/<seq>.txt, MOT format)
"""
import argparse
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

BUCKETS = ((1, 10), (11, 30), (31, 60), (61, 10 ** 9))


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    area = lambda t: (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    return w * h / (area(a) + area(b) - w * h + 1e-9)


def load_gt(path):
    gt = defaultdict(dict)  # id -> frame -> (box, vis)
    for line in open(path):
        v = line.split(",")
        if int(v[6]) != 1 or int(v[7]) != 1:
            continue
        x, y, w, h = map(float, v[2:6])
        gt[int(v[1])][int(v[0])] = ((x, y, x + w, y + h), float(v[8]))
    return gt


def load_tracks(path):
    trk = defaultdict(list)
    for line in open(path):
        v = line.split(",")
        x, y, w, h = map(float, v[2:6])
        trk[int(v[0])].append((int(float(v[1])), (x, y, x + w, y + h)))
    return trk


def match(gt, trk):
    """(gt id, frame) -> matched track id, for every frame."""
    by_frame = defaultdict(list)
    for gid, frames in gt.items():
        for f, (box, _) in frames.items():
            by_frame[f].append((gid, box))
    out = {}
    for f, items in by_frame.items():
        preds = trk.get(f, [])
        m = iou([b for _, b in items], [b for _, b in preds])
        if m.size == 0:
            continue
        cost = np.where(m >= 0.5, 1 - m, 1e6)
        for r, c in zip(*linear_sum_assignment(cost)):
            if m[r, c] >= 0.5:
                out[(items[r][0], f)] = preds[c][0]
    return out


def events(frames, hidden, visible, context, window=30):
    """Yield (first hidden frame, last hidden frame, visible frames before, visible frames after).

    People usually disappear gradually (partly covered for a few frames), so the clearly visible
    frames are searched within `window` frames before and after the hidden run, not right next to it.
    """
    fs = sorted(frames)
    vis = [frames[f][1] for f in fs]
    i = 0
    while i < len(fs):
        if vis[i] >= hidden:
            i += 1
            continue
        j = i + 1
        while j < len(fs) and vis[j] < hidden and fs[j] - fs[j - 1] <= 1:
            j += 1
        start, end = fs[i], fs[j - 1]
        before = [fs[k] for k in range(i) if start - window <= fs[k] < start and vis[k] >= visible][-context:]
        after = [fs[k] for k in range(j, len(fs)) if end < fs[k] <= end + window and vis[k] >= visible][:context]
        if len(before) == context and after:
            yield start, end, before, after
        i = j


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gt_root")
    p.add_argument("results", nargs="+")
    p.add_argument("--hidden", type=float, default=0.2)
    p.add_argument("--visible", type=float, default=0.5)
    p.add_argument("--context", type=int, default=5)
    a = p.parse_args()
    for result in a.results:
        total = {b: Counter() for b in BUCKETS}
        for f in sorted(x for x in os.listdir(result) if x.startswith("MOT") and x.endswith(".txt")):
            seq = f[:-4]
            gt = load_gt(os.path.join(a.gt_root, seq, "gt", "gt.txt"))
            matched = match(gt, load_tracks(os.path.join(result, f)))
            for gid, frames in gt.items():
                for start, end, before, after in events(frames, a.hidden, a.visible, a.context):
                    ids_before = [matched[(gid, k)] for k in before if (gid, k) in matched]
                    if not ids_before:
                        continue
                    id_before = Counter(ids_before).most_common(1)[0][0]
                    ids_after = [matched[(gid, k)] for k in after if (gid, k) in matched]
                    outcome = "missed" if not ids_after else ("kept" if ids_after[0] == id_before else "switched")
                    length = end - start + 1
                    bucket = next(b for b in BUCKETS if b[0] <= length <= b[1])
                    total[bucket][outcome] += 1
        name = os.path.basename(os.path.normpath(result))
        all_c = sum(total.values(), Counter())
        for b in BUCKETS + (None,):
            c = all_c if b is None else total[b]
            n = sum(c.values())
            label = "all" if b is None else (f"{b[0]}-{b[1]}" if b[1] < 10 ** 9 else f">{b[0] - 1}")
            if n:
                print(f"{name:28s} hidden {label:>7s} frames: events={n:4d} kept={c['kept'] / n:.2f} "
                      f"switched={c['switched'] / n:.2f} missed={c['missed'] / n:.2f}")


if __name__ == "__main__":
    main()
