#!/usr/bin/env python3
"""Run person detectors on the RK3568 NPU over image sequences; write detections + timing.

Runs on the board with rknn-toolkit-lite2 (bundled with bitpose) and librknnrt 2.5.0.
Supported model kinds (all 640x640, RGB uint8 NHWC input):
  yolo26  6 outputs: box (4,H,W) distances + cls (80,H,W) logits per stride   (NMS-free head)
  yolov8  6 outputs: box (64,H,W) DFL logits + cls (80,H,W) logits per stride
  yolo11  same head and outputs as yolov8
  yolov5  3 outputs: (255,H,W) = 3 anchors x (xywh, obj, 80 cls), sigmoid already applied
          (Rockchip rknn_model_zoo export, the model this project used before YOLO26)

Output: <out>/<model-name>/<sequence>.txt in MOTChallenge format (id = -1) and
<out>/timing.txt with the mean NPU time per model.

Usage:
  python3 eval_detectors_board.py --pylib PYLIB --runtime LIBRKNNRT --out OUT \
      --model yolo26:yolo26n.rknn --model yolov8:yolov8n.rknn --model yolov5:yolov5.rknn \
      SEQ_DIR [SEQ_DIR ...]          (SEQ_DIR contains img1/*.jpg)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

SIZE = 640
STRIDES = (8, 16, 32)
ANCHORS = ((10, 13, 16, 30, 33, 23), (30, 61, 62, 45, 59, 119), (116, 90, 156, 198, 373, 326))
PERSON = 0


def letterbox(bgr):
    h, w = bgr.shape[:2]
    s = min(SIZE / w, SIZE / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    x, y = (SIZE - nw) // 2, (SIZE - nh) // 2
    out = np.full((SIZE, SIZE, 3), 114, np.uint8)
    out[y:y + nh, x:x + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), s, x, y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def grid(h, w):
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return gx.ravel(), gy.ravel()


def decode_distances(outs, conf, dfl):
    boxes, scores = [], []
    for i, stride in enumerate(STRIDES):
        box, cls = outs[2 * i][0], outs[2 * i + 1][0]
        _, h, w = cls.shape
        score = sigmoid(cls[PERSON].ravel())
        keep = score >= conf
        if not keep.any():
            continue
        if dfl:  # (64,H,W) -> expectation over 16 bins for each of l,t,r,b
            b = box.reshape(4, 16, h * w)[:, :, keep]
            b = np.exp(b - b.max(1, keepdims=True))
            dist = (b / b.sum(1, keepdims=True) * np.arange(16, dtype=np.float32)[None, :, None]).sum(1)
        else:
            dist = box.reshape(4, h * w)[:, keep]
        gx, gy = grid(h, w)
        ax, ay = gx[keep] + 0.5, gy[keep] + 0.5
        boxes.append(np.stack([ax - dist[0], ay - dist[1], ax + dist[2], ay + dist[3]], 1) * stride)
        scores.append(score[keep])
    return boxes, scores


def decode_yolov5(outs, conf):
    boxes, scores = [], []
    for i, stride in enumerate(STRIDES):
        t = outs[i][0]
        _, h, w = t.shape
        t = t.reshape(3, 85, h * w)
        gx, gy = grid(h, w)
        for a in range(3):
            score = t[a, 4] * t[a, 5 + PERSON]
            keep = score >= conf
            if not keep.any():
                continue
            v = t[a][:, keep]
            cx = (v[0] * 2 - 0.5 + gx[keep]) * stride
            cy = (v[1] * 2 - 0.5 + gy[keep]) * stride
            bw = (v[2] * 2) ** 2 * ANCHORS[i][2 * a]
            bh = (v[3] * 2) ** 2 * ANCHORS[i][2 * a + 1]
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1))
            scores.append(score[keep])
    return boxes, scores


def nms(boxes, scores, iou_threshold, max_det=300):
    order = scores.argsort()[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


# kind -> (decoder, NMS IoU). YOLO26 is NMS-free; 0.7 only drops rare INT8 duplicates.
KINDS = {
    "yolo26": (lambda o, c: decode_distances(o, c, dfl=False), 0.7),
    "yolov8": (lambda o, c: decode_distances(o, c, dfl=True), 0.7),
    "yolo11": (lambda o, c: decode_distances(o, c, dfl=True), 0.7),
    "yolov5": (decode_yolov5, 0.45),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pylib", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", action="append", required=True, help="KIND:PATH[:NAME]")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("sequences", nargs="+")
    a = p.parse_args()

    sys.path.insert(0, a.pylib)
    from rknnlite.api import RKNNLite, rknn_runtime
    rknn_runtime.RKNNRuntime._get_rknn_api_lib_path = lambda self, *x, **k: a.runtime

    timing = []
    for spec in a.model:
        parts = spec.split(":")
        kind, path = parts[0], parts[1]
        name = parts[2] if len(parts) > 2 else os.path.splitext(os.path.basename(path))[0]
        decode, iou = KINDS[kind]
        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(path) != 0 or rknn.init_runtime() != 0:
            raise SystemExit("cannot load " + path)
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        npu_ms, post_ms, frames = 0.0, 0.0, 0
        for seq in a.sequences:
            seq_name = os.path.basename(os.path.normpath(seq))
            files = sorted(f for f in os.listdir(os.path.join(seq, "img1")) if f.endswith(".jpg"))
            with open(os.path.join(a.out, name, seq_name + ".txt"), "w") as out:
                for index, f in enumerate(files, 1):
                    img, s, px, py = letterbox(cv2.imread(os.path.join(seq, "img1", f)))
                    t0 = time.time()
                    outs = rknn.inference(inputs=[img[None]], data_format="nhwc")
                    t1 = time.time()
                    boxes, scores = decode(outs, a.conf)
                    if boxes:
                        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
                        keep = nms(boxes, scores, iou)
                        boxes, scores = boxes[keep], scores[keep]
                        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / s
                        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / s
                        for (x1, y1, x2, y2), sc in zip(boxes, scores):
                            out.write("%d,-1,%.1f,%.1f,%.1f,%.1f,%.4f,-1,-1,-1\n" % (index, x1, y1, x2 - x1, y2 - y1, sc))
                    post_ms += (time.time() - t1) * 1000
                    if index > 1:  # skip the warm-up frame for timing
                        npu_ms += (t1 - t0) * 1000
                        frames += 1
            print(name, seq_name, len(files), "frames", flush=True)
        rknn.release()
        line = "%s npu_ms=%.1f post_ms=%.1f frames=%d" % (name, npu_ms / max(1, frames), post_ms / max(1, frames + 1), frames)
        print(line, flush=True)
        timing.append(line)
    with open(os.path.join(a.out, "timing.txt"), "w") as f:
        f.write("\n".join(timing) + "\n")


if __name__ == "__main__":
    main()
