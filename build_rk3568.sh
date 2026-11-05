#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SYSROOT="${RK3568_SYSROOT:-/root/npu/ubuntu22-arm64}"
BUILD_DIR="${ROOT}/build-rk3568-gcc11"
DEPLOY_DIR="${ROOT}/deploy-rk3568"

test -f "${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake" || {
  echo "ARM64 OpenCV was not found in ${SYSROOT}" >&2
  exit 1
}
test -d /usr/include/eigen3/Eigen || {
  echo "Install Eigen first: apt-get install libeigen3-dev" >&2
  exit 1
}

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/cmake/aarch64-rk3568.cmake" \
  -DRK3568_SYSROOT="${SYSROOT}" \
  -DOpenCV_DIR="${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/opencv4" \
  -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${SYSROOT}/lib/aarch64-linux-gnu -Wl,-rpath-link,${SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${SYSROOT}/usr/lib -Wl,-rpath-link,${SYSROOT}/usr/lib/aarch64-linux-gnu/blas -Wl,-rpath-link,${SYSROOT}/usr/lib/aarch64-linux-gnu/lapack"
cmake --build "${BUILD_DIR}" --parallel
rm -rf "${DEPLOY_DIR}"
cmake --install "${BUILD_DIR}" --prefix "${DEPLOY_DIR}"

opencv_lib="${SYSROOT}/usr/lib/aarch64-linux-gnu"
for module in core imgproc; do
  cp -a "${opencv_lib}/libopencv_${module}.so"* "${DEPLOY_DIR}/lib/"
done
cp -a "${opencv_lib}/libtbb.so.2" "${DEPLOY_DIR}/lib/"
cp -a "${opencv_lib}/libstdc++.so.6"* "${DEPLOY_DIR}/lib/"
cp -a "${SYSROOT}/lib/aarch64-linux-gnu/libgcc_s.so.1" "${DEPLOY_DIR}/lib/"
cp -a "${SYSROOT}/lib/aarch64-linux-gnu/libz.so.1"* "${DEPLOY_DIR}/lib/"

echo "RK3568 deployment directory: ${DEPLOY_DIR}"
