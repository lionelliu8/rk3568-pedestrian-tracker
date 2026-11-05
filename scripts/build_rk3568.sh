#!/usr/bin/env bash
# Cross-compile for RK3568 and produce the board package.
# Output: dist/pedestrian-tracker/ and dist/pedestrian-tracker-rk3568.tar.gz
# Env: OPENCV_DIR  directory containing OpenCVConfig.cmake
#                  (default: third_party/opencv-rk3568/lib/cmake/opencv4; build it with
#                   scripts/build_opencv_rk3568.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build/rk3568"
DIST="${ROOT}/dist"
PKG="${DIST}/pedestrian-tracker"
OPENCV_DIR="${OPENCV_DIR:-${ROOT}/third_party/opencv-rk3568/lib/cmake/opencv4}"

[ -f "${OPENCV_DIR}/OpenCVConfig.cmake" ] || {
  echo "OpenCV for RK3568 not found in ${OPENCV_DIR}; run scripts/build_opencv_rk3568.sh" >&2
  exit 1
}
[ -d /usr/include/eigen3/Eigen ] || { echo "install Eigen: sudo apt install libeigen3-dev" >&2; exit 1; }

cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/cmake/toolchain-rk3568.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="${OPENCV_DIR}"
cmake --build "${BUILD}" --parallel "$(nproc)"
rm -rf "${PKG}"
cmake --install "${BUILD}" --prefix "${PKG}" > /dev/null
(cd "${PKG}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "${DIST}/pedestrian-tracker-rk3568.tar.gz" -C "${DIST}" pedestrian-tracker
echo "package: ${DIST}/pedestrian-tracker-rk3568.tar.gz"
