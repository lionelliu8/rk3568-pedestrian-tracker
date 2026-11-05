"""Select the attention form used by Ultralytics YOLO26/YOLO11 Attention blocks.

  softmax  original: attn = softmax(q^T k * scale)              (RK3568: runs on the CPU)
  sigmoid  attn = sigmoid(q^T k * scale - log N), N = key count  (RK3568: runs on the NPU)

Sigmoid attention follows Ramapuram et al., "Theory, Analysis, and Best Practices for Sigmoid
Self-Attention" (ICLR 2025); the -log N bias keeps the initial attention mass close to softmax.
It is not numerically equivalent, so a model must be trained (fine-tuned) with it.

The choice is a process-wide patch of Attention.forward: call use() before training, validation
and export, and use the same value each time. Checkpoints do not record it, so the training
scripts put it in the run name.
"""
import math

import torch
from ultralytics.nn.modules.block import Attention

_original_forward = Attention.forward


def _sigmoid_forward(self, x):
    B, C, H, W = x.shape
    N = H * W
    q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn = torch.sigmoid((q * self.scale).transpose(-2, -1) @ k - math.log(N))
    out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    return self.proj(out)


FORWARDS = {"softmax": _original_forward, "sigmoid": _sigmoid_forward}


def use(kind: str) -> None:
    if kind not in FORWARDS:
        raise ValueError(f"attention must be one of {sorted(FORWARDS)}, got {kind!r}")
    Attention.forward = FORWARDS[kind]
