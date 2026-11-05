"""Turn a COCO-pretrained detector (e.g. yolo26n.pt) into a 1-class person model.

The final classification convolutions keep only the COCO "person" row (class 0), so training
starts from the pretrained person classifier instead of a randomly initialised head.

Usage: python make_person_model.py yolo26n.pt yolo26n_person_init.pt
"""
import argparse

import torch
from torch import nn


def keep_class(conv: nn.Conv2d, index: int) -> nn.Conv2d:
    new = nn.Conv2d(conv.in_channels, 1, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight[index:index + 1])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[index:index + 1])
    return new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--class-index", type=int, default=0, help="COCO class to keep (0 = person)")
    a = p.parse_args()

    ckpt = torch.load(a.src, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    head = model.model[-1]
    for name in ("cv3", "one2one_cv3"):
        branch = getattr(head, name, None)
        if branch is None:
            continue
        for seq in branch:
            seq[-1] = keep_class(seq[-1], a.class_index)
    head.nc = 1
    head.no = 1 + head.reg_max * 4
    model.nc = 1
    model.yaml["nc"] = 1
    model.names = {0: "person"}
    torch.save({"model": model.half(), "train_args": {}, "date": None, "version": ckpt.get("version")}, a.dst)
    print(f"saved {a.dst}: 1 class (from COCO class {a.class_index})")


if __name__ == "__main__":
    main()
