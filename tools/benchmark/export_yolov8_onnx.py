"""Export YOLOv8 (DFL head, reg_max=16) with raw per-stride outputs, for the detector benchmark.

Outputs per stride s in (8, 16, 32): box_s (1, 64, H, W) DFL logits, cls_s (1, 80, H, W) logits.
Usage: python export_yolov8_onnx.py yolov8n.pt --out yolov8n_rk.onnx
"""
import argparse

import torch
from ultralytics import YOLO


class RawHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.head = model, model.model[-1]

    def forward(self, images):
        y, x = [], images
        for m in self.model.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        outs = []
        for i, j in enumerate(self.head.f):
            outs += [self.head.cv2[i](y[j]), self.head.cv3[i](y[j])]
        return tuple(outs)


p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--out", default="yolov8n_rk.onnx")
a = p.parse_args()
wrapper = RawHead(YOLO(a.weights).model.float().eval()).eval()
names = [f"{k}_s{s}" for s in (8, 16, 32) for k in ("box", "cls")]
with torch.no_grad():
    torch.onnx.export(wrapper, torch.zeros(1, 3, a.imgsz, a.imgsz), a.out, input_names=["images"],
                      output_names=names, opset_version=12, do_constant_folding=True, dynamo=False)
print("exported", a.out)
