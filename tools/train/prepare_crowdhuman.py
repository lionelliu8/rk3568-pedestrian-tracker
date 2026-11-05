#!/usr/bin/env python3
"""Convert CrowdHuman (annotation_*.odgt + extracted Images/) to YOLO labels (class 0 = person).

Uses the full-body box ("fbox"), which matches how MOT17 annotates pedestrians (occluded parts
included). Entries tagged "mask" (crowd/ignore regions) and boxes with extra.ignore == 1 are
skipped; boxes are clipped to the image.

CrowdHuman terms: non-commercial research/education only; do not redistribute the images.

Usage: prepare_crowdhuman.py CROWDHUMAN_DIR OUT_DIR [--val-images 1000]
  CROWDHUMAN_DIR holds annotation_train.odgt, annotation_val.odgt and Images/ (all zips extracted).
  OUT_DIR gets images/{train,val} (hard links when possible) and labels/{train,val}.
"""
import argparse
import json
import os
import shutil

from PIL import Image


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def convert(odgt, image_dir, out, split, limit=None):
    os.makedirs(f"{out}/images/{split}", exist_ok=True)
    os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    n_img = n_box = n_skip = 0
    with open(odgt) as f:
        for line in f:
            if limit is not None and n_img >= limit:
                break
            rec = json.loads(line)
            src = os.path.join(image_dir, rec["ID"] + ".jpg")
            if not os.path.exists(src):
                n_skip += 1
                continue
            w, h = Image.open(src).size
            rows = []
            for g in rec.get("gtboxes", []):
                if g.get("tag") != "person" or g.get("extra", {}).get("ignore", 0) == 1:
                    continue
                x, y, bw, bh = g["fbox"]
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                rows.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            link(src, f"{out}/images/{split}/{rec['ID']}.jpg")
            with open(f"{out}/labels/{split}/{rec['ID']}.txt", "w") as lab:
                lab.writelines(rows)
            n_img += 1
            n_box += len(rows)
    print(f"{split}: {n_img} images, {n_box} person boxes, {n_skip} missing images")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("crowdhuman")
    p.add_argument("out")
    p.add_argument("--val-images", type=int, default=1000, help="validation subset size (per-epoch val time)")
    a = p.parse_args()
    images = os.path.join(a.crowdhuman, "Images")
    convert(os.path.join(a.crowdhuman, "annotation_train.odgt"), images, a.out, "train")
    convert(os.path.join(a.crowdhuman, "annotation_val.odgt"), images, a.out, "val", a.val_images)


if __name__ == "__main__":
    main()
