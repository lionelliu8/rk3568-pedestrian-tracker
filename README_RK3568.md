# RK3568 YOLOv5 + DeepSORT

该版本参考原项目与 CSDN 文章的 C++ 实现路线，但针对 RK3568 做了必要调整：YOLOv5 和 OSNet ReID 都使用 RKNN NPU，DeepSORT 的卡尔曼滤波、匈牙利匹配和级联匹配运行在 CPU。RK3568 只有一个 NPU 核，因此检测与 ReID 使用单上下文顺序执行，不再使用 RK3588 的三核并发。

## WSL 构建结果

- 源码：`/root/npu3/yolov5_deepsort_rknn`
- ARM64 部署目录：`/root/npu3/yolov5_deepsort_rknn/deploy-rk3568`
- 压缩包：`/root/npu3/yolov5_deepsort_rk3568.tar.gz`
- 工具链：`aarch64-linux-gnu-g++`
- 目标系统 ABI：ARM64 glibc 2.35（已适配当前 Buildroot）

## 板端依赖

板端只需要 RK3568 的 RKNPU2 驱动、V4L2 摄像头和 DRM 显示设备；所需 OpenCV 运行库已包含在部署目录中。先检查：

```bash
ls -l /dev/video* /dev/dri/card0
```

## 运行

USB 摄像头，默认跟踪 COCO 类别 0（person）：

```bash
./run.sh --input 0 --display
```

`--display` 直接驱动 `/dev/dri/card0` 的 MIPI DSI 屏，无需 GTK、X11、Wayland 或 Qt。程序会先启动摄像头，再自动停止 QDesktop 和 Weston 并接管屏幕。数字输入 `0` 等价于 `/dev/video0`；摄像头是其他节点时可传入完整路径。

板卡重启后仍使用同一条命令；`run.sh` 会自动补齐厂家 RKISP 所需的桌面初始化顺序。

检测隔帧和类别选择：

```bash
./run.sh --input 0 --detect-interval 2 --class 0 --display
```

近距离或半身场景可降低检测阈值：

```bash
./run.sh --input /dev/video-camera0 --class 0 --conf-threshold 0.15 --display
```

`--class -1` 会跟踪全部 COCO 类别，但 OSNet 是行人 ReID 模型，默认 `--class 0` 的 ID 稳定性更可靠。通过终端按 `Ctrl+C` 退出。

## 模型来源

- YOLOv5：Rockchip `rknn_model_zoo`，明确转换为 `target_platform=rk3568`。
- ReID：TorchReID `osnet_x0_25` Market1501 权重，使用 `tools/convert_osnet_rk3568.py` 重新转换为 RK3568，输入为 RGB `128x256`，输出为 512 维特征。

WSL 中没有 RK3568 NPU 设备，因此这里只能验证模型目标标记、ARM64 编译、链接和部署文件完整性；最终推理速度与摄像头/显示必须在板端验证。
