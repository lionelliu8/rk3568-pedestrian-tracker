# 检测模型对比与视频测试（2026-11-05，RK3568 板端实测）

## 测试条件

- 数据：MOT17 训练集中的 02（拥挤街道，固定机位，1080p，600 帧）、05（手持移动拍摄，640×480，837 帧）、
  09（商业步行街，固定机位，1080p，525 帧），共 1962 帧，约 3.1 万个行人标注。
- 三个模型都是 640×640 输入、INT8 量化，都在板子 NPU 上运行（librknnrt 2.5.0），后处理方式一致：
  - YOLO26n：本项目当前使用的模型（ultralytics yolo26n.pt，COCO 预训练）
  - YOLOv8n：ultralytics yolov8n.pt，导出方式与 YOLO26n 相同
  - YOLOv5：项目之前部署的 `yolov5_rk3568.rknn`（Rockchip rknn_model_zoo 的 yolov5s_relu；
    模型里用的是 ReLU 激活，参数量约为另外两个 n 模型的 3 倍）
- 打分：`tools/benchmark/score_mot.py`，IoU ≥ 0.5。MOT17 连几乎完全被挡住的人也标注了
  （02 序列里一半标注的可见度低于 25%），所以下面分两种口径：
  “全部”= 所有行人；“可见 ≥50%”= 只统计至少一半身体可见的人，其余当作忽略区域，更接近肉眼能看到的情况。
  打分脚本的检查：MOT17 自带的 FRCNN 检测在“可见 ≥50%”口径下 AP50 = 0.89，在合理范围内。

## 检测结果（行人类，3 段序列按标注数加权）

| 模型 | NPU 推理 ms/帧 | AP50 可见≥50% | AP50 全部 | 召回率 @0.25 | 精度 @0.25 |
|---|---|---|---|---|---|
| YOLOv5s-relu（旧） | 77.6 | **0.795** | **0.492** | **0.774** | 0.757 |
| YOLOv8n | **62.0** | 0.750 | 0.456 | 0.709 | 0.782 |
| YOLO26n（当前） | 82.6 | 0.772 | 0.466 | 0.697 | **0.884** |

（NPU 时间由 Python rknnlite 测得，包含输出转换；C++ 程序里 YOLO26n 实际约 63–71 ms。召回率和精度对应的口径是“可见≥50%”。）

结论：
- 在这块板子、这类街景行人数据上，YOLO26n **没有**全面胜过旧模型。旧的 YOLOv5s 参数量约为它的 3 倍，
  AP 和召回率还是略高一点；YOLO26n 的长处是误检少（精度 0.88，旧模型 0.76）。
- 和同量级的 YOLOv8n 相比，YOLO26n 的 AP 和精度更高，但慢约 20 ms，因为它的两个注意力层 Softmax 在 RK3568 上只能放到 CPU 执行。
- 官方 COCO 指标（YOLO26n 40.9 > YOLOv8n 37.3 > YOLOv5n 28.0）对比的是同为 n 尺寸的模型，覆盖 80 类；
  板上 INT8 量化后、只看行人时，差距会缩小。
- 想要更高的召回率：可以试 yolo26s（更大），或者用行人数据微调 yolo26n。

## 跟踪结果（YOLO26n + DeepSORT，C++ 程序全流程）

| 序列 | MOTA 可见≥50% | IDF1 可见≥50% | ID 切换 | MOTA 全部 | IDF1 全部 | FPS | NPU | 跟踪 |
|---|---|---|---|---|---|---|---|---|
| 02 | 0.486 | 0.525 | 35 | 0.217 | 0.284 | 4.4 | 65 ms | 69 ms |
| 05 | 0.657 | 0.612 | 29 | 0.486 | 0.500 | 6.6 | 71 ms | 57 ms |
| 09 | 0.726 | 0.690 | 21 | 0.545 | 0.537 | 4.3 | 66 ms | 69 ms |
| 加权 | 0.584 | 0.584 | 85 | 0.334 | 0.376 | | | |

- FPS 里包含读 JPEG 的时间（1080p 约 85 ms/帧，640×480 约 22 ms/帧），接摄像头时没有这部分。
- DeepSORT 的跟踪耗时主要花在 OSNet 提取特征上（每个人跑一次 NPU），人越多越慢。
  人多的场景可以用 `--detect-interval 2`。
- 跟踪结果视频：虚拟机 `~/npu3/yolo26_work/videos/`，Windows 桌面 `RK3568_tracker_videos/`。

## 复现

```bash
# 板子上
/userdata/tracker/benchmark/run_all.sh   # 即 tools/benchmark/run_all_board.sh；跟踪 09 + 三模型检测评测，结果在 benchmark/results/
# 虚拟机上（结果拷回 ~/npu3/yolo26_work/board/results 之后）
cd ~/npu3/yolo26_work
~/rknn_venv/bin/python ~/npu3/yolov5_deepsort_rknn/tools/benchmark/score_mot.py det mot17 \
    board/results/detections/{yolo26n,yolov8n,yolov5} --min-visibility 0.5
~/rknn_venv/bin/python ~/npu3/yolov5_deepsort_rknn/tools/benchmark/score_mot.py track mot17 \
    board/results/tracks --min-visibility 0.5
```

## 补充：YOLO11n、YOLO26n 一对多检测头、sigmoid 注意力（2026-11-05）

同样的数据和打分口径（可见 ≥50%，召回率和精度取置信度 0.25 时的值）：

| 模型 | NPU ms/帧（Python） | AP50 可见≥50% | AP50 全部 | 召回率 | 精度 |
|---|---|---|---|---|---|
| YOLOv5s-relu（旧） | 77.6 | 0.795 | 0.492 | 0.774 | 0.757 |
| YOLOv8n | 62.0 | 0.750 | 0.456 | 0.709 | 0.782 |
| **YOLO11n** | 74.4 | **0.797** | **0.501** | 0.763 | 0.746 |
| YOLO26n（一对一检测头，当前用的） | 84.2 | 0.772 | 0.466 | 0.697 | **0.884** |
| YOLO26n（一对多检测头 + NMS） | 83.6 | 0.780 | 0.487 | 0.743 | 0.770 |

- YOLO11n 是 n 尺寸里 AP 最高的。它也有 1 个注意力 Softmax 在 CPU 上执行（YOLO26n 有 2 个）。
- YOLO26n 的一对一检测头不需要 NMS，误检最少，但在人挤人的场景里漏检更多。换成一对多检测头后，召回率从 0.70 升到 0.74。

逐层耗时（`tests/rknn_perf`，跑 5 次，全零输入）：

| 模型 | 注意力部分 | 总算子时间 |
|---|---|---|
| YOLO26n 原版 | 2 个 Softmax 在 CPU 上：42.7 ms | 92.0 ms |
| YOLO26n sigmoid 注意力（未训练，只测速度） | 2 个 Sigmoid 在 NPU 上：1.2 ms | 56.8 ms |

不开逐层统计实测（Python）：84.2 → 61.9 ms/帧。
其他尝试过的写法：手写 exp/max/sum、转置后对通道维做 softmax，都会被 RKNN 重新合并成 CPU Softmax；
用 sigmoid 换算 exp 再求和，因为注意力分数最大到约 17，exp 在量化时溢出成 NaN。
所以要让注意力完全在 NPU 上运行，只能改成 sigmoid 注意力，然后重新训练（微调）。
