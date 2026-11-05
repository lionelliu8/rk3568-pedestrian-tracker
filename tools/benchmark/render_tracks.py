#!/usr/bin/env python3
"""Draw MOT-format tracks onto an image sequence and encode an H.264 MP4 (needs ffmpeg).

Usage: render_tracks.py SEQ_DIR TRACKS.txt OUT.mp4 [--fps 25] [--width 1280] [--title TEXT]
"""
import argparse
import os
import subprocess
from collections import defaultdict

import cv2


def color(i):
    return ((37 * i + 67) % 205 + 50, (17 * i + 113) % 205 + 50, (29 * i + 31) % 205 + 50)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("seq")
    p.add_argument("tracks")
    p.add_argument("out")
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--title", default="")
    a = p.parse_args()

    boxes = defaultdict(list)
    for line in open(a.tracks):
        v = line.split(",")
        boxes[int(v[0])].append((int(float(v[1])), *map(float, v[2:6])))
    files = sorted(f for f in os.listdir(os.path.join(a.seq, "img1")) if f.endswith(".jpg"))
    first = cv2.imread(os.path.join(a.seq, "img1", files[0]))
    scale = min(1.0, a.width / first.shape[1])
    w, h = int(first.shape[1] * scale) // 2 * 2, int(first.shape[0] * scale) // 2 * 2
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                            "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for n, f in enumerate(files, 1):
        img = cv2.resize(cv2.imread(os.path.join(a.seq, "img1", f)), (w, h))
        for tid, x, y, bw, bh in boxes.get(n, []):
            p1 = (int(x * scale), int(y * scale))
            p2 = (int((x + bw) * scale), int((y + bh) * scale))
            cv2.rectangle(img, p1, p2, color(tid), 2)
            cv2.putText(img, str(tid), (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color(tid), 2)
        label = f"{a.title}  frame {n}/{len(files)}  tracks {len(boxes.get(n, []))}"
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
