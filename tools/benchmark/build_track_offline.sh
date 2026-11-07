#!/usr/bin/env bash
# Build the host (x86) offline tracker used for tracker comparison and tuning.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "${ROOT}/build/host"
g++ -O2 -std=c++17 -Wall -Wextra -I"${ROOT}/src/common" -I"${ROOT}/src/mot" -I/usr/include/eigen3 \
  "${ROOT}/tools/benchmark/track_offline.cpp" "${ROOT}/src/mot/assignment.cpp" "${ROOT}/src/mot/bytetrack.cpp" \
  "${ROOT}/src/mot/ocsort.cpp" -o "${ROOT}/build/host/track_offline"
echo "${ROOT}/build/host/track_offline"
