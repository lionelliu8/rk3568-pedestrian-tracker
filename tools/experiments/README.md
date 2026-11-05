# 注意力 Softmax 实验（2026-11-05）

问题：YOLO26n 的两个注意力 Softmax 在 RK3568 上被 RKNN 放到 CPU 上执行，每帧约 27–43 ms。

| 脚本 | 内容 | 结果 |
|---|---|---|
| `probe_attention.py` | 导出两个变体：`expsig`（用 sigmoid 换算 exp、用矩阵乘法求和，数学上和原版等价）和 `sigmoid`（sigmoid 注意力，数学上和原版不同） | expsig：量化时溢出成 NaN，转换失败；sigmoid：全部算子都在 NPU 上，84 → 62 ms/帧，但需要重新训练 |
| `export_yolo26_o2m.py` | 用 YOLO26 的一对多检测头导出（需要 NMS） | AP50 0.772 → 0.780，召回率 0.70 → 0.74 |

早期的两种改写已经从导出脚本里删掉（结果见 docs/BENCHMARK_20261105.md）：
转置后对通道维做 softmax、手写 exp/max/sum，都被 RKNN 重新合并成 CPU Softmax。

FP16（不量化）对照：`probe_yolo26n_fp16.rknn` 里 Softmax 仍然在 CPU 上（FLOAT16 CPU），
说明放到 CPU 是因为 RK3568 NPU 不支持这个算子，和量化无关。

这几个脚本需要和 `tools/export_yolo26_onnx.py` 一起运行（训练机，ultralytics 环境），例如：
`python tools/experiments/probe_attention.py`（工作目录里要有 yolo26n.pt）。
