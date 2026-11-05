# RK3568 RGA resize test

The RGA package contains deploy-rk3568/ at the archive root.
Stop the existing tracker before replacing its files:

```sh
killall yolov5_deepsort
tar -xzf /userdata/yolov5_deepsort_rk3568_rga.tar.gz -C /userdata
/userdata/deploy-rk3568/run.sh --input /dev/video-camera0 --class 0 --display
```

Expected log: `YOLO resize backend: RGA`.
If RGA rejects the buffers or driver, the application logs the error and uses
CPU resize for the rest of that run. This does not handle a missing librga.so;
always deploy the included lib/ directory with the executable.

CPU comparison, with the same scene and arguments:

```sh
killall yolov5_deepsort
RGA_RESIZE=0 /userdata/deploy-rk3568/run.sh --input /dev/video-camera0 --class 0 --display
```

Compare the existing `YOLO profile ms: preprocess=...` log and FPS after warmup.
Check person box alignment, colors and image proportions in both runs.
Only YOLO BGR resize is accelerated. Letterbox, color conversion, OSNet resize
and display processing remain on the CPU. RGA interpolation can differ slightly
from OpenCV; board-side correctness and speed have not yet been measured.
