#!/usr/bin/env python3
"""Add MOT17 training sequences to a YOLO person dataset (class 0).

Keeps ground-truth pedestrians (class 1, consider flag 1) and every --step-th frame, because
consecutive video frames are nearly identical. Do not add the sequences used for testing
(MOT17-02/05/09 in docs/BENCHMARK_20261105.md).

MOT17 is licensed CC BY-NC-SA 3.0 (non-commercial).

Usage: prepare_mot17.py OUT_DIR SEQ_DIR [SEQ_DIR ...] [--step 3]
"""
import argparse
import configparser
import os
import shutil
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("sequences", nargs="+")
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--split", default="train")
    a = p.parse_args()
    os.makedirs(f"{a.out}/images/{a.split}", exist_ok=True)
    os.makedirs(f"{a.out}/labels/{a.split}", exist_ok=True)
    for seq in a.sequences:
        name = os.path.basename(os.path.normpath(seq)).replace("-FRCNN", "")
        if name in {"MOT17-02", "MOT17-05", "MOT17-09"}:
            raise SystemExit(f"{name} is a test sequence; do not train on it")
        info = configparser.ConfigParser()
        info.read(os.path.join(seq, "seqinfo.ini"))
        w, h = int(info["Sequence"]["imWidth"]), int(info["Sequence"]["imHeight"])
        boxes = defaultdict(list)
        for line in open(os.path.join(seq, "gt", "gt.txt")):
            v = line.split(",")
            if int(v[6]) != 1 or int(v[7]) != 1:
                continue
            x, y, bw, bh = map(float, v[2:6])
            x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                boxes[int(v[0])].append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        frames = sorted(f for f in os.listdir(os.path.join(seq, "img1")) if f.endswith(".jpg"))
        n_img = n_box = 0
        for f in frames[::a.step]:
            idx = int(f[:-4])
            stem = f"{name}_{idx:06d}"
            dst = f"{a.out}/images/{a.split}/{stem}.jpg"
            if not os.path.exists(dst):
                shutil.copy(os.path.join(seq, "img1", f), dst)
            with open(f"{a.out}/labels/{a.split}/{stem}.txt", "w") as lab:
                lab.writelines(boxes.get(idx, []))
            n_img += 1
            n_box += len(boxes.get(idx, []))
        print(f"{name}: {n_img} images, {n_box} person boxes")


if __name__ == "__main__":
    main()
