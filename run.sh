#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="${ROOT}/lib:${LD_LIBRARY_PATH:-}"

for argument in "$@"; do
  if [ "${argument}" = "--display" ]; then
    if ! pidof weston >/dev/null; then
      /etc/init.d/S49weston start >/dev/null 2>&1
      sleep 2
    fi
    if ! pidof QDesktop >/dev/null; then
      /etc/init.d/S50QDesktop start >/dev/null 2>&1
      sleep 3
    fi
    break
  fi
done

exec "${ROOT}/yolov5_deepsort" "$@"
