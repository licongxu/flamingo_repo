#!/usr/bin/env bash
# Wait until no processes match any of the given grep patterns.
# Usage: wait_for_flamingo_builds.sh 'build_halo_catalogue' 'build_y_map.py'
set -euo pipefail

POLL_SEC="${FLAMINGO_BUILD_POLL_SEC:-120}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 PATTERN [PATTERN...]" >&2
  exit 1
fi

while true; do
  found=0
  for pat in "$@"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
      found=1
      echo "$(date -Is) waiting for processes matching: ${pat}"
      break
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "$(date -Is) no matching build processes; proceeding."
    break
  fi
  sleep "$POLL_SEC"
done
