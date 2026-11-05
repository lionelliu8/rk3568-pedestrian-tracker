#!/usr/bin/env python3
"""Convert the raw-head YOLO26 ONNX (see export_yolo26_onnx.py) to an INT8 RKNN model for RK3568.

Calibration images are letterboxed exactly like the C++ runtime (pad value 114) before
quantization, so the INT8 ranges match what the NPU sees on the board.

Usage:
  python convert_yolo26_rknn.py yolo26n_rk.onnx --calib DIR --out model/yolo26n_rk3568.rknn \
      [--imgsz 640] [--max-calib 200] [--check-image bus.jpg]
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN

STRIDES = (8, 16, 32)


def letterbox(bgr, size, pad=114):
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), pad, np.uint8)
    out[y:y + nh, x:x + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, x, y


def build_dataset(calib_dir, work, size, limit):
    files = sorted(p for p in Path(calib_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    files = files[:limit]
    if not files:
        raise SystemExit(f"no calibration images in {calib_dir}")
    lines = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        lb, *_ = letterbox(img, size)
        dst = work / f"calib_{i:04d}.png"
        cv2.imwrite(str(dst), lb)
        lines.append(str(dst))
    dataset = work / "dataset.txt"
    dataset.write_text("\n".join(lines) + "\n")
    print(f"calibration images: {len(lines)}")
    return dataset


def decode(outputs, conf=0.25):
    """Same decode as src/detector/yolo26_detector.cpp (used only for the self-check)."""
    dets = []
    for i, stride in enumerate(STRIDES):
        box, cls = outputs[2 * i][0], outputs[2 * i + 1][0]
        score = 1 / (1 + np.exp(-cls))
        best, cid = score.max(0), score.argmax(0)
        for gy, gx in zip(*np.where(best >= conf)):
            l, t, r, b = box[:, gy, gx]
            ax, ay = gx + 0.5, gy + 0.5
            dets.append(((ax - l) * stride, (ay - t) * stride, (ax + r) * stride, (ay + b) * stride,
                         float(best[gy, gx]), int(cid[gy, gx])))
    return sorted(dets, key=lambda d: -d[4])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx")
    p.add_argument("--calib", required=True, help="directory of representative images")
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-calib", type=int, default=200)
    p.add_argument("--check-image", help="run the simulator on this image and print detections")
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dataset = build_dataset(a.calib, Path(tmp), a.imgsz, a.max_calib)
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3568",
                    quantized_algorithm="normal", optimization_level=3)
        if rknn.load_onnx(model=a.onnx) != 0:
            raise SystemExit("load_onnx failed")
        if rknn.build(do_quantization=True, dataset=str(dataset)) != 0:
            raise SystemExit("build failed")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(a.out) != 0:
            raise SystemExit("export_rknn failed")
        print("exported", a.out)

        if a.check_image:
            img = cv2.imread(a.check_image)
            lb, scale, px, py = letterbox(img, a.imgsz)
            rknn.init_runtime()
            outs = rknn.inference(inputs=[cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)[None]], data_format="nhwc")
            for x1, y1, x2, y2, s, c in decode(outs)[:20]:
                print(f"class={c:2d} score={s:.2f} box=({(x1 - px) / scale:.0f},{(y1 - py) / scale:.0f},"
                      f"{(x2 - px) / scale:.0f},{(y2 - py) / scale:.0f})")
        rknn.release()


if __name__ == "__main__":
    main()
