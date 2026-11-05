#!/bin/sh
# Board launcher. With --display the desktop is stopped so the tracker can take over the DRM
# screen, and started again when the tracker exits.
# Example: ./run.sh --input /dev/video-camera0 --display
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

display=0
for argument in "$@"; do
  [ "${argument}" = "--display" ] && display=1
done

if [ "${display}" = 1 ]; then
  /etc/init.d/S50QDesktop stop >/dev/null 2>&1
  /etc/init.d/S49weston stop >/dev/null 2>&1
  sleep 1
fi

# Keep the shell alive on Ctrl-C so the desktop is restored; the tracker still gets the signal.
trap : INT TERM
"${ROOT}/pedestrian_tracker" "$@"
status=$?

if [ "${display}" = 1 ] && [ "${RESTORE_DESKTOP:-1}" = 1 ]; then
  /etc/init.d/S49weston start >/dev/null 2>&1
  /etc/init.d/S50QDesktop start >/dev/null 2>&1
fi
exit "${status}"
