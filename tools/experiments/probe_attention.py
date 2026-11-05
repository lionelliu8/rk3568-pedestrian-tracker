"""Export YOLO26n variants whose attention avoids the CPU-only Softmax on RK3568 (speed/placement probe).

  expsig : exact softmax. exp(x) = s / (1 - s) with s = sigmoid(x); the row sum is a MatMul with ones.
           No max subtraction, so it relies on |scores| staying small (checked on real images below).
  sigmoid: sigmoid attention, attn = sigmoid(scores - log N). Different maths, needs fine-tuning.
"""
import math
import sys

import torch
from ultralytics.nn.modules.block import Attention

sys.argv = sys.argv[:1]
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import export_yolo26_onnx as ex  # tools/export_yolo26_onnx.py


def forward_expsig(self, x):
    B, C, H, W = x.shape
    N = H * W
    q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2)
    scores = (q * self.scale).transpose(-2, -1) @ k
    s = torch.sigmoid(scores)
    e = s / (1 - s)
    total = e @ torch.ones(N, 1)
    attn = e / total
    return self.proj((v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W)))


def forward_sigmoid(self, x):
    B, C, H, W = x.shape
    N = H * W
    q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn = torch.sigmoid((q * self.scale).transpose(-2, -1) @ k - math.log(N))
    return self.proj((v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W)))


original = Attention.forward
stats = []


def forward_measure(self, x):
    B, C, H, W = x.shape
    N = H * W
    q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2)
    stats.append(((q * self.scale).transpose(-2, -1) @ k).abs().max().item())
    return original(self, x)


from ultralytics import YOLO
from ultralytics.utils import ASSETS
model = YOLO("yolo26n.pt")
Attention.forward = forward_measure
for img in ("bus.jpg", "zidane.jpg"):
    model.predict(str(ASSETS / img), verbose=False)
print("max |attention score| on sample images:", max(stats))

for name, fwd in (("expsig", forward_expsig), ("sigmoid", forward_sigmoid)):
    Attention.forward = fwd
    wrapper = ex.RawHead(YOLO("yolo26n.pt").model.float().eval()).eval()
    names = [f"{k}_s{s}" for s in (8, 16, 32) for k in ("box", "cls")]
    dummy = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        if name == "expsig":
            new = wrapper(dummy)
            Attention.forward = original
            ref = ex.RawHead(YOLO("yolo26n.pt").model.float().eval()).eval()(dummy)
            Attention.forward = fwd
            print("expsig max abs diff vs original:", max((a - b).abs().max().item() for a, b in zip(new, ref)))
        torch.onnx.export(wrapper, torch.zeros(1, 3, 640, 640), f"yolo26n_{name}.onnx", input_names=["images"],
                          output_names=names, opset_version=12, do_constant_folding=True, dynamo=False)
    print("exported", name)
