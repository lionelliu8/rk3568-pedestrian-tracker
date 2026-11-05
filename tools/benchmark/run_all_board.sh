#!/bin/sh
# Board: track MOT17-09 with the C++ tracker, then compare the three detectors on all test sequences.
# Expects /userdata/tracker/{pedestrian-tracker,testdata/MOT17-xx,benchmark/{models,pylib}}.
# Installed on the board as /userdata/tracker/benchmark/run_all.sh.
T=/userdata/tracker
mkdir -p $T/benchmark/results/tracks
cd $T/pedestrian-tracker && ./pedestrian_tracker --input $T/testdata/MOT17-09/img1 --mot-output $T/benchmark/results/tracks/MOT17-09.txt > /tmp/trk09.log 2>&1
cd $T/benchmark && python3 eval_detectors_board.py --pylib pylib --runtime $T/pedestrian-tracker/lib/librknnrt.so --out results/detections \
  --model yolo26:models/yolo26n_rk3568.rknn:yolo26n --model yolov8:models/yolov8n_rk3568.rknn:yolov8n \
  --model yolov5:models/yolov5_rk3568.rknn:yolov5 $T/testdata/MOT17-02 $T/testdata/MOT17-05 $T/testdata/MOT17-09 > /tmp/eval.log 2>&1
echo ALL_DONE >> /tmp/eval.log
