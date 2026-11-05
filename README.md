# RK3568 行人检测与跟踪（YOLO26 + DeepSORT）

摄像头画面 → YOLO26n 检测（NPU，INT8）→ DeepSORT 跟踪（OSNet ReID 在 NPU 上，卡尔曼滤波和匈牙利匹配在 CPU 上）→ 可选 DRM 屏幕显示。

- 板卡：ATK-DLRK3568（Buildroot，glibc 2.35，RKNPU 驱动 0.8.2）
- 运行时：librknnrt 2.5.0（随包附带，板子自带的 1.3.0 太旧）
- 开发机：Ubuntu 22.04 虚拟机（用户 llx），所有源码、构建和部署都在这里进行
- 训练机：Windows GPU 主机（用户 y，RX 7900 XT，ultralytics 8.4），只在需要训练或导出时使用

## 目录结构

```
CMakeLists.txt
cmake/toolchain-rk3568.cmake     aarch64 交叉编译工具链
src/
  main.cpp                       主流程：采集 → 检测 → 跟踪 → 显示
  app/        options.*          命令行参数
              overlay.*          画框、标签、FPS
  common/     detect_box.h       检测框/跟踪框结构
              stopwatch.h        计时
  detector/   yolo26_detector.*  YOLO26 RKNN 推理：letterbox 预处理（RGA 缩放）+ 解码 + NMS
  io/         v4l2_camera.*      V4L2 mmap 采集（YUYV/NV12/NV21/BGR/RGB → BGR）
              drm_display.*      DRM dumb buffer 全屏显示
  tracker/                       DeepSORT（基于上游 JackWoo0831/yolov5_deepsort_rknn）
tests/detector_selftest.cpp      不需要摄像头的板端检测自检和测速
tools/
  export_yolo26_onnx.py          YOLO26 .pt → 适配 RKNN 的 ONNX（在训练机上运行）
  convert_yolo26_rknn.py         ONNX → INT8 RKNN，并带仿真器自检（在虚拟机 ~/rknn_venv 中运行）
  convert_osnet_rk3568.py        OSNet ReID → RKNN
  image_to_bgr.py                图片 → 自检用的原始 BGR 文件
scripts/
  build_opencv_rk3568.sh         编译静态 OpenCV（core + imgproc）
  build_rk3568.sh                交叉编译并打包，产物在 dist/
  deploy_rk3568.sh               通过 SSH 部署到板子
  run.sh                         板端启动脚本
model/                           yolo26n_rk3568.rknn、osnet_x0_25_market_rk3568.rknn、COCO 标签
third_party/                     rknn（2.5.0）、rga
docs/UPSTREAM_README.md          上游项目原始说明
```

## 构建

```bash
sudo apt install g++-aarch64-linux-gnu cmake libeigen3-dev
scripts/build_opencv_rk3568.sh            # 只需执行一次，生成 third_party/opencv-rk3568
scripts/build_rk3568.sh                   # 生成 dist/pedestrian-tracker-rk3568.tar.gz
```

也可以用 `OPENCV_DIR=<目录>` 指定已有的静态 OpenCV，例如复用 6D 项目编好的：
`OPENCV_DIR=~/6D/cross/opencv-install/lib/cmake/opencv4 scripts/build_rk3568.sh`。

C++ 运行库（libstdc++/libgcc）静态链接，OpenCV 也是静态库，所以包里只需要附带 `librknnrt.so` 和 `librga.so`。

## 部署与运行

```bash
scripts/deploy_rk3568.sh root@192.168.110.166   # 解压到 /userdata/pedestrian-tracker
```

板端：

```sh
cd /userdata/pedestrian-tracker
./run.sh --display                         # 摄像头 /dev/video-camera0，结果画到屏幕
./run.sh --max-frames 300                  # 不显示，只打印 FPS 和各阶段耗时
./run.sh --class -1 --conf-threshold 0.3   # 跟踪全部 COCO 类别
./run.sh --detect-interval 2               # 每 2 帧检测一次，中间帧用卡尔曼预测
```

使用 `--display` 时，`run.sh` 会先停掉桌面（weston/QDesktop），退出后再恢复；设置 `RESTORE_DESKTOP=0` 则不恢复。
摄像头同一时间只能被一个程序使用，运行前要先关掉其他占用摄像头的程序（例如 bitpose 预览、系统相机）。

每秒输出一行，例如：

```
frames=120 tracks=3 fps=… | ms: pre=… npu=… post=… track=… display=…
```

不接摄像头时检查检测是否正常：

```sh
# 在虚拟机上：tools/image_to_bgr.py bus.jpg bus.bgr  → 输出 810 1080
LD_LIBRARY_PATH=lib tests/detector_selftest model/yolo26n_rk3568.rknn bus.bgr 810 1080 30
PERF_DETAIL=1 LD_LIBRARY_PATH=lib tests/detector_selftest ...   # 打印逐层 NPU/CPU 耗时
```

## 更换或重新训练 YOLO26 模型

1. 训练机（`ssh gpu`，环境 `D:\bitpose_train\venv`）：
   ```
   python tools/export_yolo26_onnx.py yolo26n.pt --imgsz 640 --out yolo26n_rk.onnx
   ```
   导出脚本会去掉检测头里的解码和 top-k，只保留 6 个原始输出（3 个尺度 × 框/类别）。
2. 虚拟机：
   ```
   ~/rknn_venv/bin/python tools/convert_yolo26_rknn.py yolo26n_rk.onnx \
       --calib <代表性图片目录> --out model/yolo26n_rk3568.rknn --check-image bus.jpg
   ```
   校准图片最好来自实际场景，或者用 COCO 图片；脚本会按运行时的方式做 letterbox。
3. 重新执行 `scripts/build_rk3568.sh` 和部署。

自己训练的模型如果类别不同，换掉 `model/coco_80_labels_list.txt`，并用 `--class` 指定要跟踪的类别。
DeepSORT 的 ReID 是行人模型，跟踪行人以外的类别时 ID 稳定性会差一些。

## 检测输出格式

RKNN 模型有 1 个输入（640×640 RGB uint8，NHWC）和 6 个输出：每个步长（8/16/32）各一个
`box (1,4,H,W)`（到锚点中心的左/上/右/下距离，单位是网格）和 `cls (1,80,H,W)`（logit）。
C++ 端在 INT8 量化域里直接和阈值比较（不用对每个锚点算 exp），只对通过阈值的点做反量化，
框坐标为 `(anchor ∓ dist) × stride`。YOLO26 是一对一检测头，本身不需要 NMS；
代码里保留了一个 IoU 0.7 的轻量 NMS，用来去掉量化偶尔带来的重复框。

## 性能（板端实测，2026-11-05，bus.jpg，640×640 INT8）

| 阶段 | 耗时 |
|---|---|
| 预处理（letterbox） | 约 10 ms（RGA 要求宽度 16 对齐，不对齐时走 CPU） |
| NPU 推理 | 约 92 ms |
| 后处理（80 类全扫描；只跟踪行人时更低） | 约 6–10 ms |

`PERF_DETAIL=1` 的逐层统计显示：两个注意力层（model.10、model.22）里的 Softmax 被 RKNN 放到 CPU 上执行，
每帧约 27 ms，占推理时间的 27%。尝试过两种等价改写（转置后对通道维做 softmax；手写 exp/max/sum），
RKNN 2.5.0 都会把它们重新合并成 CPU Softmax，所以目前保留原结构。
如需更快，可以把 `--imgsz` 降到 480 或 512（推理时间大致和像素数成正比，但小目标的召回率会下降），
或者每 2 帧检测一次（`--detect-interval 2`）。
