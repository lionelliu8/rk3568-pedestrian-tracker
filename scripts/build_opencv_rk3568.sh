#!/usr/bin/env bash
# Build a minimal static OpenCV 4.5.5 (core + imgproc + imgcodecs with JPEG/PNG) for RK3568 into third_party/opencv-rk3568.
# Usage: scripts/build_opencv_rk3568.sh [opencv-4.5.5.tar.gz]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARBALL="${1:-${OPENCV_TARBALL:-$HOME/6D/cross/opencv-4.5.5.tar.gz}}"
WORK="${ROOT}/build/opencv"
PREFIX="${ROOT}/third_party/opencv-rk3568"

[ -f "${TARBALL}" ] || { echo "OpenCV source tarball not found: ${TARBALL}" >&2; exit 1; }
mkdir -p "${WORK}"
[ -d "${WORK}/opencv-4.5.5" ] || tar -xzf "${TARBALL}" -C "${WORK}"

cmake -S "${WORK}/opencv-4.5.5" -B "${WORK}/build" \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/cmake/toolchain-rk3568.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF \
  -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF \
  -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DWITH_PROTOBUF=OFF \
  -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF \
  -DBUILD_ZLIB=ON -DCPU_BASELINE=NEON -DCPU_DISPATCH= -DENABLE_PRECOMPILED_HEADERS=OFF
cmake --build "${WORK}/build" --parallel "$(nproc)"
# OpenCV 4.5.5 exports quirc/ADE targets even when their modules are not built;
# build the small archives so find_package(OpenCV) can validate them.
cmake --build "${WORK}/build" --target quirc ade --parallel "$(nproc)"
cmake --install "${WORK}/build"
echo "OpenCV installed to ${PREFIX}"
