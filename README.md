# RK3568 行人检测与跟踪（YOLO26n-person + ByteTrack）

摄像头画面（或图片序列）→ YOLO26n 行人检测（NPU，INT8）→ ByteTrack 跟踪（CPU，每帧约 0.3 ms）
→ 可选屏幕显示（RGA 硬件缩放和旋转）。也可以用 `--tracker ocsort` 或 `--tracker deepsort` 切换跟踪算法。
读帧、NPU 推理、跟踪和显示分别在三个线程上流水执行，图像的缩放/颜色转换/旋转全部交给 RGA，**摄像头实时约 24 FPS**。

默认检测模型 `model/yolo26n_person_rk3568.rknn` 是在官方 YOLO26n 基础上做的两处改动：
1. 注意力层的 Softmax 换成 **sigmoid 注意力**（RK3568 的 NPU 不支持 Softmax，原版这一步要退回 CPU）；
2. 用行人数据（CrowdHuman + MOT17）**微调成只检测行人的单类别模型**。

板端实测（MOT17 测试序列，只统计至少一半身体可见的行人，INT8）：

| 配置 | FPS | 检测 AP50 | 跟踪 MOTA | 跟踪 IDF1 |
|---|---|---|---|---|
| 官方 YOLO26n + DeepSORT（最初的版本） | ~7 | 0.772 | 0.584 | 0.584 |
| YOLO26n-person + DeepSORT | ~8 | 0.845 | 0.656 | 0.618 |
| **YOLO26n-person + ByteTrack（默认）** | **24（摄像头）** | **0.845** | **0.687** | **0.705** |

完整对比（YOLOv5s / YOLOv8n / YOLO11n / 各种改写）见 `docs/BENCHMARK_20261105.md`。

- 板卡：ATK-DLRK3568（Buildroot，glibc 2.35，RKNPU 驱动 0.8.2）
- 运行时：librknnrt 2.5.0（随包附带，板子自带的 1.3.0 太旧）
- 开发机：Ubuntu 22.04 虚拟机（用户 llx），所有源码、构建和部署都在这里进行
- 训练机：Windows GPU 主机（用户 y，RX 7900 XT，ROCm + ultralytics 8.4）

## 目录结构

```
CMakeLists.txt
cmake/toolchain-rk3568.cmake     aarch64 交叉编译工具链
src/
  main.cpp                       主流程：读帧 → 检测 → 跟踪 → 输出/显示
  app/        options.*          命令行参数
              overlay.*          画框、标签、FPS
              mot_writer.*       结果写成 MOTChallenge 格式
  common/     detect_box.h       检测框/跟踪框结构
              stopwatch.h        计时
  detector/   yolo26_detector.*  YOLO26 RKNN 推理：letterbox（RGA 缩放）+ INT8 域解码 + NMS，类别数自动识别
  io/         frame_source.*     输入接口：摄像头或图片目录
              v4l2_camera.*      V4L2 mmap 采集（YUYV/NV12/NV21/BGR/RGB → BGR）
              image_sequence.*   按文件名顺序读取图片目录（视频测试用）
              drm_display.*      DRM dumb buffer 全屏显示
  mot/        bytetrack.*        ByteTrack（默认）
              ocsort.*           OC-SORT
              assignment.*       匈牙利算法、IoU
  app/        multi_tracker.*    三种跟踪算法的统一接口（--tracker）
  common/     bounded_queue.h    流水线线程之间的队列
  tracker/                       DeepSORT（基于上游 JackWoo0831/yolov5_deepsort_rknn）
tests/
  detector_selftest.cpp          不需要摄像头的检测自检和测速
  rknn_perf.cpp                  任意 RKNN 模型的逐层耗时（看算子在 NPU 还是 CPU 上）
tools/
  export_yolo26_onnx.py          .pt → 适配 RKNN 的 ONNX（训练机上运行）
  convert_yolo26_rknn.py         ONNX → INT8 RKNN，带仿真器自检（虚拟机 ~/rknn_venv）
  convert_osnet_rk3568.py        OSNet ReID → RKNN
  image_to_bgr.py                图片 → 自检用的原始 BGR 文件
  train/                         行人微调：sigmoid 注意力、单类别初始化、数据转换、训练脚本（见其 README）
  benchmark/                     板端多模型对比、MOT17 打分、结果渲染成视频
  experiments/                   Softmax 改写实验、一对多检测头实验（见其 README）
scripts/
  build_opencv_rk3568.sh         编译静态 OpenCV（core + imgproc + imgcodecs）
  build_rk3568.sh                交叉编译并打包，产物在 dist/
  deploy_rk3568.sh               通过 SSH 部署到板子
  run.sh                         板端启动脚本
model/
  yolo26n_person_rk3568.rknn     默认：行人单类别，sigmoid 注意力
  person_labels.txt              默认标签
  yolo26n_rk3568.rknn            官方 YOLO26n，COCO 80 类（需要检测其他物体时用）
  coco_80_labels_list.txt
  osnet_x0_25_market_rk3568.rknn DeepSORT 的 ReID 模型
third_party/                     rknn（2.5.0）；rga（板子自带的 librga 1.3.1，只用于编译，不打包）
docs/                            BENCHMARK_20261105.md（测试记录）、UPSTREAM_README.md（上游说明）
```

## 构建

```bash
sudo apt install g++-aarch64-linux-gnu cmake libeigen3-dev
scripts/build_opencv_rk3568.sh            # 只需执行一次，生成 third_party/opencv-rk3568
scripts/build_rk3568.sh                   # 生成 dist/pedestrian-tracker-rk3568.tar.gz
```

C++ 运行库和 OpenCV 都是静态链接，包里只附带 `librknnrt.so`。librga 用板子自带的
`/usr/lib/librga.so.2`：它必须和内核的 RGA 驱动版本匹配，原项目附带的 librga 1.6.0 与这块板子的驱动不兼容，
RGA 一直调用失败、悄悄退回 CPU（2026-11-07 修复）。

## 部署与运行

```bash
scripts/deploy_rk3568.sh root@192.168.110.166   # 解压到 /userdata/tracker/pedestrian-tracker
```

板端：

```sh
cd /userdata/tracker/pedestrian-tracker
./run.sh --display                         # 摄像头 /dev/video-camera0，结果画到屏幕
./run.sh --max-frames 300                  # 不显示，只打印 FPS 和各阶段耗时
./run.sh --tracker ocsort                  # 换成 OC-SORT（或 deepsort）
./run.sh --conf-threshold 0.6              # ByteTrack 高分阈值（默认 0.5，新轨迹需要 +0.1）
./run.sh --no-pipeline                     # 不用流水线，逐帧顺序处理（对比耗时用）
./run.sh --input /userdata/tracker/testdata/MOT17-09/img1 --display   # 用录好的视频帧代替摄像头
./run.sh --input <图片目录> --mot-output tracks.txt                     # 跟踪结果存成 MOTChallenge 格式
# 检测行人以外的物体：换回官方 80 类模型
./run.sh --model model/yolo26n_rk3568.rknn --labels model/coco_80_labels_list.txt --class -1
```

`run.sh` 在程序运行期间把 NPU（900 MHz）和 CPU（1.99 GHz）都固定在最高频率，退出后恢复原来的
`rknpu_ondemand` / `interactive`。默认的动态调频在摄像头输入时升不上去：NPU 单帧要 50 ms 而不是 41 ms。
`NPU_PERFORMANCE=0`、`CPU_PERFORMANCE=0` 分别关闭。连续运行 5 分钟芯片温度稳定在约 71°C，
低于 75°C 的降频阈值（室温下、无散热片实测；装进封闭外壳时建议加散热片）。
使用 `--display` 时，`run.sh` 会先停掉桌面（weston/QDesktop），退出后再恢复；设置 `RESTORE_DESKTOP=0` 则不恢复。
摄像头同一时间只能被一个程序使用，运行前要先关掉其他占用摄像头的程序（例如 bitpose 预览、系统相机）。

每秒输出一行：

```
frames=120 tracks=3 fps=… | ms: read=… pre=… npu=… post=… track=… display=…
```

## 性能（板端实测，2026-11-07，默认模型 + ByteTrack，经 run.sh 运行）

| 输入 | FPS | NPU | 预处理 | 跟踪 | 显示 |
|---|---|---|---|---|---|
| 摄像头 1280×720 | **24** | 41 ms | 3.6 ms | 0.1 ms | — |
| 摄像头 + 屏幕显示 | 约 21–24 | | | | 9 ms |
| 1080p 图片序列（MOT17-09） | 28 | 35 ms | 8 ms | 0.3 ms | — |
| 1080p 图片序列 + 屏幕显示 | 27.5 | 36 ms | 8 ms | 0.3 ms | 14 ms |

三个阶段并行，帧率由最慢的一段（NPU）决定。CPU 上只剩：取摄像头帧、填灰边、`rknn_run` 里驱动/数据搬运那部分、
解码检测框、ByteTrack、画框。像素处理（摄像头 NV12→BGR、缩放 + BGR→RGB + 放进 640×640、屏幕缩放旋转）都在 RGA 上。
摄像头输入时只转换最新的一帧，来不及处理的帧直接还给驱动（日志里的 `dropped`）。

提速过程（摄像头）：DeepSORT 7 → ByteTrack 15.8 → 流水线 18.3 → NPU 固定最高频率 22 → RGA 颜色转换 + CPU 固定最高频率 24。

## 视频测试（不需要现场找行人）

`--input` 给一个图片目录时，按文件名顺序逐帧读取，相当于播放一段视频。
板上有 3 段 MOT17 测试数据：`/userdata/tracker/testdata/MOT17-{02,05,09}`；标注在虚拟机的 `~/npu3/yolo26_work/mot17/`。

```bash
tools/benchmark/score_mot.py track <gt根目录> <结果目录> --min-visibility 0.5   # MOTA / IDF1
tools/benchmark/score_mot.py det   <gt根目录> <结果目录> --min-visibility 0.5   # AP50 / 召回率 / 精度
tools/benchmark/render_tracks.py <序列目录> tracks.txt out.mp4                   # 画框生成 MP4
```

`--min-visibility 0.5` 只统计至少一半身体可见的人。MOT17 连几乎完全被遮挡的人也标注了，不加这个参数时所有模型的分数都会偏低。

## 为什么要改注意力层

YOLO26 和 YOLO11 的注意力层里有 Softmax，RK3568 的 NPU 不支持这个算子，RKNN 会把它放到 CPU 上执行
（YOLO26n 有 2 个，每帧约 27–43 ms；FP16 不量化也一样，说明和量化无关）。
试过的等价改写要么被 RKNN 重新合并成 Softmax，要么在量化时溢出（见 `tools/experiments/README.md`）。
最后的做法是改成 sigmoid 注意力：`attn = sigmoid(q·k·scale − log N)`
（Ramapuram et al., *Theory, Analysis, and Best Practices for Sigmoid Self-Attention*, ICLR 2025）。
它在 NPU 上只要 1.2 ms，但数学上和 Softmax 不等价，必须重新训练。和同样数据训练的 Softmax 版本相比，
精度只低 0.005（AP50 0.845 vs 0.850），推理快 23 ms。

## 跟踪算法选择

在 MOT17-05 上选参数，在 MOT17-02/09 和遮挡测试上比较（`tools/benchmark/track_offline.cpp` 离线评测，
板端 C++ 结果与离线结果一致）。可见 ≥50%：

| 算法 | MOTA | IDF1 | ID 切换 | 真实遮挡后保持 ID | 柱子遮挡后保持 ID | 跟踪耗时 |
|---|---|---|---|---|---|---|
| DeepSORT | 0.645 | 0.614 | 71 | 51% | 45% | 45–130 ms |
| **ByteTrack** | **0.672** | **0.698** | **33** | **70%** | **52%** | 0.3 ms |
| OC-SORT | 0.615 | 0.600 | 58 | 53% | 45% | 0.5 ms |

OC-SORT 论文里它在 MOT17 上和 ByteTrack 基本持平（HOTA 63.2 对 63.1），明显的优势在动作非线性的 DanceTrack 上。
这个检测模型的分数整体偏低，ByteTrack 会再用低分框（0.1–0.5）维持已有轨迹，所以对它更合适；
OC-SORT 换了几组参数也没有超过 ByteTrack（见 `docs/BENCHMARK_20261105.md`）。

## 训练（重新训练或换数据）

详见 `tools/train/README.md`，流程：

1. 数据：`prepare_crowdhuman.py`、`prepare_mot17.py` 转成 YOLO 格式（全身框，单类别）。
   这次用了 CrowdHuman 15000 张 + MOT17-04/10/11/13（每 3 帧取 1 帧，共 1118 帧），验证集为 CrowdHuman val 1000 张。
   MOT17-02/05/09 只用于测试，不能参与训练。
2. `make_person_model.py`：从官方 yolo26n.pt 生成单类别模型，分类层直接复用 COCO 的 person 权重。
3. `train_person.py --attention sigmoid`：100 轮，batch 32，640×640，其他是 ultralytics 默认值；
   在 RX 7900 XT 上约 17 小时（两个版本并行）。
4. `export_yolo26_onnx.py --attention sigmoid` 导出，`convert_yolo26_rknn.py` 转换。
   校准图片要用行人场景（这次用 CrowdHuman val 150 张 + MOT17 训练序列 50 张）。

**数据许可**：CrowdHuman 只允许非商业研究和教学使用，且图片不得再分发；MOT17 是 CC BY-NC-SA 3.0。
所以本仓库不包含任何训练图片，用这两个数据集训练出的 `yolo26n_person_rk3568.rknn` 也只能用于非商业用途。
商用需要换成有商用许可的数据重新训练。

## 检测输出格式

RKNN 模型有 1 个输入（640×640 RGB uint8，NHWC）和 6 个输出：每个步长（8/16/32）各一个
`box (1,4,H,W)`（到锚点中心的左/上/右/下距离，单位是网格）和 `cls (1,C,H,W)`（logit，C 为类别数，行人模型 C=1）。
C++ 端在 INT8 量化域里直接和阈值比较（不用对每个锚点算 exp），只对通过阈值的点做反量化，
框坐标为 `(anchor ∓ dist) × stride`。YOLO26 是一对一检测头，本身不需要 NMS；
代码里保留了一个 IoU 0.7 的轻量 NMS，用来去掉量化偶尔带来的重复框。
