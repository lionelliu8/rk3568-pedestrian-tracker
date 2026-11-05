#!/usr/bin/env python3
"""Convert an image to raw BGR bytes for tests/detector_selftest. Prints WIDTH HEIGHT."""
import sys

import cv2

img = cv2.imread(sys.argv[1])
if img is None:
    raise SystemExit(f"cannot read {sys.argv[1]}")
img.tofile(sys.argv[2])
print(img.shape[1], img.shape[0])
