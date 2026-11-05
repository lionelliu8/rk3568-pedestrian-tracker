"""Export an Ultralytics YOLO26 detection model to an RKNN-friendly ONNX graph.

The stock export decodes boxes and runs top-k inside the graph, which the RK3568 NPU
handles poorly. This script keeps only the convolutional part of the NMS-free
one-to-one head and returns its raw tensors, one pair per stride:

    box_s8  (1, 4,  H/8,  W/8)    distances left, top, right, bottom (grid units)
    cls_s8  (1, nc, H/8,  W/8)    class logits (sigmoid applied on the CPU)
    box_s16, cls_s16, box_s32, cls_s32

Splitting box and class tensors gives each its own INT8 scale. The C++ decoder in
src/detector/yolo26_detector.cpp reproduces the dropped decode step.

Usage: python export_yolo26_onnx.py yolo26n.pt --imgsz 640 --out yolo26n_rk.onnx
"""
import argparse
import os
import sys

import torch
from ultralytics import YOLO


class RawHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head = model.model[-1]
        if getattr(self.head, "one2one_cv2", None) is None:
            raise SystemExit("model has no one-to-one head; YOLO26 detection weights are required")
        if self.head.reg_max != 1:
            raise SystemExit(f"expected reg_max=1 (YOLO26), got {self.head.reg_max}")

    def forward(self, images):
        # Run every layer except the head, keeping the outputs the head consumes.
        y, x = [], images
        for m in self.model.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        feats = [y[j] for j in self.head.f]
        outs = []
        for i, f in enumerate(feats):
            outs.append(self.head.one2one_cv2[i](f))
            outs.append(self.head.one2one_cv3[i](f))
        return tuple(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--out", default="yolo26n_rk.onnx")
    p.add_argument("--attention", choices=["softmax", "sigmoid"], default="softmax",
                   help="must match the attention the weights were trained with (tools/train/attention.py)")
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(here, "train"), here]  # repo layout, or files copied side by side
    import attention
    attention.use(a.attention)

    model = YOLO(a.weights).model.float().eval()
    wrapper = RawHead(model).eval()
    strides = [int(s) for s in wrapper.head.stride.tolist()]
    names = [f"{k}_s{s}" for s in strides for k in ("box", "cls")]
    dummy = torch.zeros(1, 3, a.imgsz, a.imgsz)
    with torch.no_grad():
        for n, t in zip(names, wrapper(dummy)):
            print(n, tuple(t.shape))
        torch.onnx.export(wrapper, dummy, a.out, input_names=["images"], output_names=names,
                          opset_version=a.opset, do_constant_folding=True, dynamo=False)
    try:
        import onnx
        from onnxsim import simplify
        m, ok = simplify(onnx.load(a.out))
        if ok:
            onnx.save(m, a.out)
    except ImportError:
        pass
    print("exported", a.out, "classes", wrapper.head.nc, "strides", strides)


if __name__ == "__main__":
    main()
