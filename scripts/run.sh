#!/bin/sh
# Board launcher.
# - Locks the NPU clock at its maximum while the tracker runs and restores the governor afterwards.
#   With a camera the default rknpu_ondemand governor does not ramp the NPU up between frames:
#   inference takes ~54 ms instead of ~44 ms (18 vs 22 FPS). NPU_PERFORMANCE=0 disables this.
# - Also locks the CPU at its maximum (1.99 GHz; part of rknn_run runs on the CPU, and the
#   interactive governor keeps it lower under a light camera load). CPU_PERFORMANCE=0 disables.
# - With --display the desktop is stopped so the tracker can take over the DRM screen, and
#   started again when the tracker exits (RESTORE_DESKTOP=0 keeps it stopped).
# Example: ./run.sh --input /dev/video-camera0 --display
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
NPU_GOVERNOR=/sys/class/devfreq/fde40000.npu/governor
CPU_GOVERNOR=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor  # all four A55 cores

display=0
for argument in "$@"; do
  [ "${argument}" = "--display" ] && display=1
done

if [ "${display}" = 1 ]; then
  /etc/init.d/S50QDesktop stop >/dev/null 2>&1
  /etc/init.d/S49weston stop >/dev/null 2>&1
  sleep 1
fi

old_npu=""
old_cpu=""
if [ "${NPU_PERFORMANCE:-1}" = 1 ] && [ -w "${NPU_GOVERNOR}" ]; then
  old_npu="$(cat "${NPU_GOVERNOR}")"
  echo performance > "${NPU_GOVERNOR}"
fi
if [ "${CPU_PERFORMANCE:-1}" = 1 ] && [ -w "${CPU_GOVERNOR}" ]; then
  old_cpu="$(cat "${CPU_GOVERNOR}")"
  echo performance > "${CPU_GOVERNOR}"
fi

# Keep the shell alive on Ctrl-C so the settings are restored; the tracker still gets the signal.
trap : INT TERM
"${ROOT}/pedestrian_tracker" "$@"
status=$?

[ -n "${old_npu}" ] && echo "${old_npu}" > "${NPU_GOVERNOR}"
[ -n "${old_cpu}" ] && echo "${old_cpu}" > "${CPU_GOVERNOR}"
if [ "${display}" = 1 ] && [ "${RESTORE_DESKTOP:-1}" = 1 ]; then
  /etc/init.d/S49weston start >/dev/null 2>&1
  /etc/init.d/S50QDesktop start >/dev/null 2>&1
fi
exit "${status}"
