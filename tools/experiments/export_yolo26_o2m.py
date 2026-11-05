"""YOLO26 raw export using the one-to-many head (needs NMS), for comparison with the default one-to-one head."""
import sys
import torch
from ultralytics import YOLO
sys.argv = sys.argv[:1]
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import export_yolo26_onnx as ex  # tools/export_yolo26_onnx.py


class O2M(ex.RawHead):
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


model = YOLO("yolo26n.pt").model.float().eval()
assert model.model[-1].cv2 is not None, "one-to-many head missing"
w = O2M(model).eval()
names = [f"{k}_s{s}" for s in (8, 16, 32) for k in ("box", "cls")]
with torch.no_grad():
    torch.onnx.export(w, torch.zeros(1, 3, 640, 640), "yolo26n_o2m.onnx", input_names=["images"], output_names=names,
                      opset_version=12, do_constant_folding=True, dynamo=False)
print("exported yolo26n_o2m.onnx")
