#!/usr/bin/env python3
"""Paint fixed occluders (pillars / a railing) onto a MOT sequence and update the ground truth.

Each ground-truth box's visibility becomes  vis * (1 - covered fraction of the box), so
score_mot.py --min-visibility and occlusion_reid.py treat people behind the occluders correctly.
Occluders are flat grey with vertical shading so they look like pillars, not like people.

Usage:
  make_occlusion_test.py SEQ_DIR OUT_DIR --pillar 0.30:0.40 --pillar 0.62:0.70 [--rail 0.72:1.0]
  (pillar = x range, rail = y range, both as fractions of the image size)

The output keeps the MOT layout (img1/, gt/gt.txt, seqinfo.ini). Copy img1/ to the board and
the whole directory to the VM for scoring.
"""
import argparse
import os
import shutil

import cv2
import numpy as np


def span(text):
    a, b = (float(x) for x in text.split(":"))
    return a, b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq")
    p.add_argument("out")
    p.add_argument("--pillar", type=span, action="append", default=[])
    p.add_argument("--rail", type=span, action="append", default=[])
    p.add_argument("--quality", type=int, default=92)
    a = p.parse_args()

    files = sorted(f for f in os.listdir(os.path.join(a.seq, "img1")) if f.endswith(".jpg"))
    h, w = cv2.imread(os.path.join(a.seq, "img1", files[0])).shape[:2]
    mask = np.zeros((h, w), bool)
    paint = np.zeros((h, w, 3), np.uint8)
    for x0, x1 in a.pillar:
        c0, c1 = int(x0 * w), int(x1 * w)
        shade = 90 + 60 * np.sin(np.linspace(0, np.pi, c1 - c0))  # rounded-pillar shading
        paint[:, c0:c1] = shade[None, :, None].astype(np.uint8)
        mask[:, c0:c1] = True
    for y0, y1 in a.rail:
        r0, r1 = int(y0 * h), int(y1 * h)
        shade = 80 + 50 * np.sin(np.linspace(0, np.pi, r1 - r0))
        paint[r0:r1, :] = shade[:, None, None].astype(np.uint8)
        mask[r0:r1, :] = True

    os.makedirs(os.path.join(a.out, "img1"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "gt"), exist_ok=True)
    shutil.copy(os.path.join(a.seq, "seqinfo.ini"), a.out)
    for f in files:
        img = cv2.imread(os.path.join(a.seq, "img1", f))
        img[mask] = paint[mask]
        cv2.imwrite(os.path.join(a.out, "img1", f), img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])

    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    def covered(x1, y1, x2, y2):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        s = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return s / ((x2 - x1) * (y2 - y1))
    with open(os.path.join(a.seq, "gt", "gt.txt")) as src, open(os.path.join(a.out, "gt", "gt.txt"), "w") as dst:
        for line in src:
            v = line.strip().split(",")
            x, y, bw, bh = map(float, v[2:6])
            v[8] = f"{float(v[8]) * (1 - covered(x, y, x + bw, y + bh)):.4f}"
            dst.write(",".join(v) + "\n")
    print(f"wrote {len(files)} frames to {a.out}; occluders cover {mask.mean():.1%} of the image")


if __name__ == "__main__":
    main()
