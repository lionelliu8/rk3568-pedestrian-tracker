#!/usr/bin/env bash
# Copy the package to the board over SSH and unpack it to /userdata/tracker/pedestrian-tracker.
# Usage: scripts/deploy_rk3568.sh [user@host]   (default: root@192.168.110.166)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${1:-${RK3568_BOARD:-root@192.168.110.166}}"
TARBALL="${ROOT}/dist/pedestrian-tracker-rk3568.tar.gz"

[ -f "${TARBALL}" ] || { echo "build first: scripts/build_rk3568.sh" >&2; exit 1; }
scp "${TARBALL}" "${BOARD}:/tmp/"
ssh "${BOARD}" 'set -e
  killall pedestrian_tracker 2>/dev/null || true
  rm -rf /userdata/tracker/pedestrian-tracker
  mkdir -p /userdata/tracker
  gunzip -c /tmp/pedestrian-tracker-rk3568.tar.gz | tar -x -C /userdata/tracker  # BusyBox tar has no -z
  cd /userdata/tracker/pedestrian-tracker && sha256sum -c SHA256SUMS >/dev/null && echo "deployed to /userdata/tracker/pedestrian-tracker"'
