# 行人检测微调（训练机：ssh gpu，Windows + ROCm，RX 7900 XT）

把本目录的文件和 `tools/export_yolo26_onnx.py` 拷到训练机的 `D:\pedestrian_train\`（放在同一个目录里），
Python 环境用 `D:\bitpose_train\venv`（ultralytics 8.4）。

0. 准备数据（在服务器或虚拟机上执行，然后把结果拷到训练机）：
   `prepare_crowdhuman.py <CrowdHuman目录> <输出目录>`，使用全身框，验证集取 1000 张；
   `prepare_mot17.py <输出目录> MOT17-04 MOT17-10 MOT17-11 MOT17-13`，每 3 帧取 1 帧，不能加入测试用的 02/05/09。
   两份数据都只允许非商业研究使用，CrowdHuman 的图片不能再分发，不要提交到 git。
1. 生成单类别（行人）初始模型，分类层直接复用 COCO 的 person 权重：
   `python make_person_model.py yolo26n.pt yolo26n_person_init.pt`
   未训练时在 coco128 的行人上 mAP50 = 0.70，说明权重复制正确。
2. 训练（`--attention` 可选 softmax 或 sigmoid，见 `attention.py`）：
   `python train_person.py data.yaml --attention sigmoid --name y26n_sigmoid`
   长时间训练要用计划任务 `pedestrian_train` 启动 `run_train.ps1`，ssh 断开不会把它杀掉。
3. 导出 ONNX，`--attention` 必须和训练时一致：
   `python export_yolo26_onnx.py runs/y26n_sigmoid/weights/best.pt --attention sigmoid --out y26n_sigmoid.onnx`
4. 在虚拟机上转换：`tools/convert_yolo26_rknn.py`。转出来的单类别模型可以直接用于 C++ 程序，检测器会自动识别类别数。

训练机已知问题：
- Python 进程做完工作后会在退出阶段卡死，持续占用 CPU，所以脚本最后用 `os._exit(0)` 强制退出。
- 在 ssh 会话里用 Start-Process 启动的进程，会话结束时会被一起杀掉。
- 训练机上的 Python 访问 HTTPS 不稳定，数据集要先在服务器上下载好再拷过去。

冒烟测试（2026-11-06，coco128 里的 102 张行人图训练 3 轮，26 张验证）：

| 模型 | person mAP50 |
|---|---|
| 单类别初始模型，Softmax，未训练 | 0.702 |
| 直接换成 sigmoid，未训练 | 0.000 |
| Softmax 训练 3 轮 | 0.697 |
| sigmoid 训练 3 轮 | 0.448 |

sigmoid 冒烟模型在板子上：YOLO26 单类别，NPU 43.6 ms，后处理 1.6 ms，没有算子在 CPU 上（`tests/detector_selftest`）。
原版 80 类 YOLO26n 在同一测试里 NPU 92 ms，后处理 9.9 ms。

正式训练结果（2026-11-06）：见 docs/BENCHMARK_20261105.md 最后一节。sigmoid 版本已成为默认模型 `model/yolo26n_person_rk3568.rknn`。
